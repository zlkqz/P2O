import torch
import requests
import json
from typing import List, Dict, Any
from collections import defaultdict

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("memory_feedback")
class MemoryFeedbackRewardManager:
    """
    Memory Feedback Reward Manager
    
    这个reward manager在计算reward后，会将使用的memory_id和对应的成功/失败状态
    发送给memory server的feedback API进行更新。
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", 
                 seperate_write_reward=False, memory_feedback_url=None, 
                 success_threshold=0) -> None:
        """
        Initialize the MemoryFeedbackRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
            seperate_write_reward: Whether to separate write reward from answer reward.
            memory_feedback_url: URL of the memory server feedback API (e.g., "http://localhost:8000/memory/feedback/batch")
            success_threshold: Threshold above which a reward is considered successful (default: 0.5)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.seperate_write_reward = seperate_write_reward
        self.memory_feedback_url = memory_feedback_url
        self.success_threshold = success_threshold

    def __call__(self, data: DataProto, return_dict=False):
        """
        Memory feedback aware reward computation.
        
        Computes rewards and sends feedback to memory server about which memories
        were used and whether the task was successful.
        
        Args:
            data (DataProto): Input data containing prompts, responses, and metadata
            return_dict (bool): Whether to return additional information as a dictionary
            
        Returns:
            torch.Tensor or dict: Reward tensor or dictionary with reward_tensor and reward_extra_info
        """

        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        already_print_data_sources = {}
        
        if 'answer_boundaries' not in data.meta_info:
            data.meta_info['answer_boundaries'] = {}

        feedback_data = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            if self.seperate_write_reward:
                answer_boundaries = data.meta_info.get('answer_boundaries', {})
                answer_boundary = answer_boundaries.get(i, None)
                write_rewards = data.meta_info.get('write_rewards', {})
                write_reward = write_rewards.get(i, 0.0)

                if answer_boundary is not None and answer_boundary < valid_response_length:
                    reward_tensor[i, answer_boundary] = score
                    reward_tensor[i, valid_response_length - 1] = write_reward
                else:
                    reward_tensor[i, valid_response_length - 1] = score
                    data.meta_info['answer_boundaries'][i] = valid_response_length - 1
            else:
                reward_tensor[i, valid_response_length - 1] = score

            used_memory_ids = data.meta_info.get('used_memory_ids', None)
            if used_memory_ids is not None and i < len(used_memory_ids) and used_memory_ids[i]:
                is_success = score > self.success_threshold
                
                for memory_id in used_memory_ids[i]:
                    feedback_data.append({
                        "memory_id": memory_id,
                        "success": is_success,
                        "task_id": f"sample_{i}",
                        "details": f"score={score:.3f}, threshold={self.success_threshold}"
                    })

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"Sample {i}: score={score:.3f}, data_source={data_source}")
                if i < len(used_memory_ids) and used_memory_ids[i]:
                    print(f"  Used memory_ids: {used_memory_ids[i]}")

        if self.memory_feedback_url is not None:
            self._send_feedback_to_memory_server(feedback_data)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info
            }
        else:
            return reward_tensor

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