import torch
import requests
import json
import re
from typing import List, Dict, Any
from collections import defaultdict

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("memory_feedback_with_tool")
class MemoryFeedbackRewardManagerWithTool:
    """
    Memory Feedback Reward Manager
    
    这个reward manager在计算reward后，会将使用的memory_id和对应的成功/失败状态
    发送给memory server的feedback API进行更新。
    """

    def __init__(self, num_examine, compute_score=None, reward_fn_key="data_source", 
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

    def valid_answer_correct(self, answer, golden_answer):
        def extract_winner(text):
            text = text.strip()
            normalized_text = re.sub(r"\s+", " ", text)
            
            pattern = r"^\[\[\s*([A-Za-z])\s*([<>])\s*([A-Za-z])\s*\]\]$"
            match = re.match(pattern, normalized_text)
            if not match:
                return None

            left_symbol = match.group(1).upper()
            operator_symbol = match.group(2)
            right_symbol = match.group(3).upper()

            if left_symbol not in {"A", "B"} or right_symbol not in {"A", "B"}:
                return None

            if operator_symbol == "<":
                return right_symbol.strip()
            elif operator_symbol == ">":
                return left_symbol.strip()
            else:
                return None
        answer_winner = extract_winner(answer)
        target_winner = extract_winner(golden_answer)
        if answer_winner is not None and target_winner is not None and answer_winner == target_winner:
            return True
        else:
            return False

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
            
            answer_score = self.valid_answer_correct(extracted_data["get_golden_answer_args"], extracted_data["next_message_content"])
            answer_score = 1. if answer_score else 0.
            return {
                "format_score": 1.,
                "answer_score": answer_score,
                "score": answer_score,
            }

        except (json.JSONDecodeError, KeyError, IndexError):
            return False

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

        for i, (conversation, used_mids) in enumerate(zip(full_conversations, used_memory_ids)):
            score = 1.0 if self.validate_conversation(conversation) else 0.0
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