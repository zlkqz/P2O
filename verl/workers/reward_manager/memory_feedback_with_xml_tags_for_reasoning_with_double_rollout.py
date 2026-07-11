import torch
import requests
import json
import re
from typing import List, Dict, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

from verl.utils.reward_score.webins import ScoringVerifier


@register("memory_feedback_with_xml_tags_for_reasoning_with_double_rollout")
class MemoryFeedbackRewardManagerWithXMLTagsForReasoningWithDoubleRollout:
    """
    Memory Feedback Reward Manager
    
    这个reward manager在计算reward后，会将使用的memory_id和对应的成功/失败状态
    发送给memory server的feedback API进行更新。
    """

    def __init__(self, num_examine, verifier_base_url, compute_score=None, reward_fn_key="data_source", 
                 seperate_write_reward=False, memory_feedback_url=None, 
                 success_threshold=0) -> None:
        """
        Initialize the MemoryFeedbackRewardManager instance.

        Args:
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
            seperate_write_reward: Whether to separate write reward from answer reward.
            memory_feedback_url: URL of the memory server feedback API (e.g., "http://localhost:8000/memory/feedback/batch")
            success_threshold: Threshold above which a reward is considered successful (default: 0.5)
        """
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.seperate_write_reward = seperate_write_reward
        self.memory_feedback_url = memory_feedback_url
        self.success_threshold = success_threshold

        self.verifier = ScoringVerifier(verifier_base_url=verifier_base_url)

    def valid_answer_correct(self, question, answer, golden_answer):
        try:
            score_result = self.verifier.compute_score(
                solution_str=answer,
                question=question,
                ground_truth=golden_answer,
                return_details=True
            )
            
            if isinstance(score_result, dict):
                score = score_result.get('score', 0)
            else:
                score = score_result
                
        except Exception as e:
            print(f"Error computing score: {e}")
            score = 0.

        return score

    def validate_conversation(self, conversation, question, golden_answer):
        """
        验证conversation是否满足指定的规则
        
        Args:
            conversation (list): 包含消息的列表，每个消息都有role等字段
        """
        zero_score_dict = {
            "format_score": 0.,
            "answer_score": 0.,
            "score": 0.,
        }

        if len(conversation) != 2 or conversation[1]["role"] != "assistant":
            return zero_score_dict
        assistant_content = conversation[1]["content"]
        try:    
            pattern = r'<(search|my_answer|used_memory)>(.*?)</\1>'
            all_matches = re.findall(pattern, assistant_content, re.DOTALL)
            if all_matches:
                all_patterns = [(match[0], match[1].strip()) for match in all_matches]
            else:
                return zero_score_dict
            
            actions = [action for action, _ in all_patterns]
            if len(actions) < 2:
                return zero_score_dict
            if set(actions[-2:]) != {"my_answer", "used_memory"}:
                return zero_score_dict
            if any(action != "search" for action in actions[:-2]):
                return zero_score_dict
            
            answer_content = ""
            for action, content in all_patterns:
                if action == "my_answer":
                    answer_content = content
            answer_score = self.valid_answer_correct(question, answer_content, golden_answer)
            return {
                "format_score": 1.,
                "answer_score": answer_score,
                "score": answer_score,
            }
            
        except (json.JSONDecodeError, KeyError, IndexError):
            return zero_score_dict

    def validate_no_memory_conversation(self, no_memory_conversation, question, golden_answer):
        """
        验证无需记忆检索的对话是否满足指定的规则
        """
        zero_score_dict = {
            "format_score": 0.,
            "answer_score": 0.,
            "score": 0.,
        }
        no_memory_content = no_memory_conversation[1]["content"]
        try:
            pattern = r'<(my_answer)>(.*?)</\1>'
            all_matches = re.findall(pattern, no_memory_content, re.DOTALL)
            if all_matches:
                all_patterns = [(match[0], match[1].strip()) for match in all_matches]
            else:
                return zero_score_dict
            if len(all_patterns) != 1:
                return zero_score_dict
            answer_content = all_patterns[0][1]
            answer_score = self.valid_answer_correct(question, answer_content, golden_answer)
            return {
                "format_score": 1.,
                "answer_score": answer_score,
                "score": answer_score,
            }
        except (json.JSONDecodeError, KeyError, IndexError):
            return zero_score_dict

    def __call__(self, data, return_dict=False):
        """
        Memory feedback aware reward computation.
        
        Computes rewards and sends feedback to memory server about which memories
        were used and whether the task was successful.
            
        Returns:
            torch.Tensor or dict: Reward tensor or dictionary with reward_tensor and reward_extra_info
        """
        feedback_data = []

        with_memory_results = self.validate_conversations_parallel(data)
        no_memory_results = self.validate_no_memory_conversation_parallel(data)
        used_memory_ids = data["meta_info"].get("used_memory_ids", None)

        for i, (wm_result, nm_result, used_mids) in enumerate(zip(with_memory_results, no_memory_results, used_memory_ids)):
            wm_score = wm_result["score"]
            nm_score = nm_result["score"]
            if used_mids:
                wm_is_success = wm_score > self.success_threshold
                nm_is_success = nm_score > self.success_threshold
                for memory_id in used_mids:
                    feedback_data.append({
                        "memory_id": memory_id,
                        "with_memory_success": wm_is_success,
                        "no_memory_success": nm_is_success,
                        "task_id": f"sample_{i}",
                        "details": f"with_memory_score={wm_score:.3f}, no_memory_score={nm_score:.3f}, threshold={self.success_threshold}"
                    })
            print(f"sample_{i}   with_memory_score={wm_score:.3f}, no_memory_score={nm_score:.3f}, threshold={self.success_threshold}")
        if self.memory_feedback_url is not None:
            self._send_feedback_to_memory_server(feedback_data)

        return with_memory_results

    def validate_conversations_parallel(self, data):
        before_answer_conversations = data.get("before_answer_conversations", None)
        questions = data["meta_info"].get("questions", None)
        golden_answers = data["meta_info"].get("golden_answers", None)
        assert questions is not None and golden_answers is not None

        conversations = before_answer_conversations
        results = [None] * len(conversations)
        with ThreadPoolExecutor(max_workers=min(len(conversations), 64)) as executor:
            future_to_index = {
                executor.submit(self.validate_conversation, conversation, question, golden_answer): i
                for i, (question, conversation, golden_answer) in enumerate(zip(questions, conversations, golden_answers))
            }
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    print(f"Error validating conversation at index {i}: {e}")
                    results[i] = {
                        "format_score": 0.,
                        "answer_score": 0.,
                        "score": 0.,
                    }
        return results

    def validate_no_memory_conversation_parallel(self, data):
        questions = data["meta_info"].get("questions", None)
        golden_answers = data["meta_info"].get("golden_answers", None)
        assert questions is not None and golden_answers is not None
        no_memory_conversations = data.get("no_memory_conversations", None)
        assert no_memory_conversations
        results = [None] * len(no_memory_conversations)
        with ThreadPoolExecutor(max_workers=min(len(no_memory_conversations), 64)) as executor:
            future_to_index = {
                executor.submit(self.validate_no_memory_conversation, no_memory_conversation, question, golden_answer): i
                for i, (question, no_memory_conversation, golden_answer) in enumerate(zip(questions, no_memory_conversations, golden_answers))
            }
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    print(f"Error validating no memory conversation at index {i}: {e}")
                    results[i] = {
                        "format_score": 0.,
                        "answer_score": 0.,
                        "score": 0.,
                    }
        return results

    def _send_feedback_to_memory_server(self, feedback_data: List[Dict[str, Any]]):
        """
        发送feedback数据到memory server
        
        Args:
            feedback_data: List of feedback dictionaries containing memory_id, success, etc.
        """
        payload = {
            "feedbacks": feedback_data
        }
        
        print(f"Sending feedback to memory server: {len(feedback_data)} items")
        response = requests.post(
            self.memory_feedback_url,
            json=payload,
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Memory feedback sent successfully: {result.get('updated_count', 0)} memories updated")
        else:
            print(f"Failed to send memory feedback: {response.status_code}, {response.text}")