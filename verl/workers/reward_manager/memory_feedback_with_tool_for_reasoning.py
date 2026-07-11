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


@register("memory_feedback_with_tool_for_reasoning")
class MemoryFeedbackRewardManagerWithToolForReasoning:
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

    def validate_conversation(self, conversation):
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

        question = ""
        for msg in conversation:
            if msg.get("role") == "user":
                content = msg.get("content")
                content = content.split("Question: ")
                if len(content) != 0:
                    question = content[-1].strip()
                break
        if question == "":
            question = "[No Question Provided]"

        assistant_messages = [msg for msg in conversation if msg.get("role") == "assistant" and msg.get("tool_calls") is not None]
        
        if not assistant_messages:
            return zero_score_dict
        
        for msg in assistant_messages:
            tool_calls = msg.get("tool_calls", [])
            if len(tool_calls) > 1:
                for tc in tool_calls:
                    if tc.get("function", {}).get("name") != "search_memory":
                        return zero_score_dict
                
        write_to_memory_count = 0
        get_golden_answer_count = 0
        write_to_memory_msg_index = -1
        get_golden_answer_msg_index = -1
        search_memory_msg_indices = []
        
        for i, msg in enumerate(assistant_messages):
            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                continue
            if len(tool_calls) == 1:
                func_name = tool_calls[0].get("function", {}).get("name")
                if func_name == "write_to_memory":
                    write_to_memory_count += 1
                    write_to_memory_msg_index = i
                elif func_name == "get_golden_answer":
                    get_golden_answer_count += 1
                    get_golden_answer_msg_index = i
                elif func_name == "search_memory":
                    search_memory_msg_indices.append(i)
            else:
                search_memory_msg_indices.append(i)
        
        if write_to_memory_count != 1 or write_to_memory_msg_index != len(assistant_messages) - 1:
            return zero_score_dict
        
        if get_golden_answer_count != 1:
            return zero_score_dict
        
        if get_golden_answer_msg_index == -1 or write_to_memory_msg_index == -1:
            return zero_score_dict
        for idx in search_memory_msg_indices:
            if idx > get_golden_answer_msg_index:
                return zero_score_dict
        if not (get_golden_answer_msg_index < write_to_memory_msg_index):
            return zero_score_dict
        
        get_golden_answer_msg = assistant_messages[get_golden_answer_msg_index]
        
        try:
            arguments_str = get_golden_answer_msg["tool_calls"][0]["function"]["arguments"]
            arguments_dict = json.loads(arguments_str)
            answer = arguments_dict.get("answer", "")
            
            get_golden_answer_original_index = -1
            for i, msg in enumerate(conversation):
                if (msg.get("role") == "assistant" and 
                    msg.get("tool_calls") and
                    len(msg["tool_calls"]) == 1 and
                    msg["tool_calls"][0]["function"]["name"] == "get_golden_answer"):
                    get_golden_answer_original_index = i
                    break
            
            if get_golden_answer_original_index == -1 or get_golden_answer_original_index + 1 >= len(conversation):
                return zero_score_dict
            
            next_message = conversation[get_golden_answer_original_index + 1]
            next_content = next_message.get("content", "")
            
            extracted_data = {
                "get_golden_answer_args": answer,
                "next_message_content": next_content
            }
            
            answer_score = self.valid_answer_correct(question, extracted_data["get_golden_answer_args"], extracted_data["next_message_content"])
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

        full_conversations = data.get("full_conversations", None)
        used_memory_ids = data["meta_info"].get("used_memory_ids", None)

        results = [None] * len(full_conversations)
        with ThreadPoolExecutor(max_workers=min(len(full_conversations), 32)) as executor:
            future_to_index = {
                executor.submit(self.validate_conversation, conversation): i
                for i, conversation in enumerate(full_conversations)
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

        for i, (result, used_mids) in enumerate(zip(results, used_memory_ids)):
            score = result["score"]
            if used_mids:
                is_success = score > self.success_threshold
                for memory_id in used_mids:
                    feedback_data.append({
                        "memory_id": memory_id,
                        "success": is_success,
                        "task_id": f"sample_{i}",
                        "details": f"score={score:.3f}, threshold={self.success_threshold}"
                    })
            print(f"sample_{i} score={score:.3f}, threshold={self.success_threshold}")
        if self.memory_feedback_url is not None:
            self._send_feedback_to_memory_server(feedback_data)

        return None

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