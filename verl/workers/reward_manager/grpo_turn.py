from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

import numpy as np


@register("grpo_turn")
class GRPOTurnRewardManager:
    """The reward manager adapted for GRPO_TURN advantage estimator.
    
    This reward manager extends the functionality of NaiveRewardManager by adding
    support for GRPO_TURN, which requires turn_split_indices for multi-turn advantage estimation.
    
    Key features:
    1. Maintains full compatibility with existing reward computation logic
    2. Automatically generates turn_split_indices for GRPO_TURN advantage estimator
    3. Supports both standard reward mode and separate write reward mode
    4. Uses the same interface and patterns as other reward managers in verl
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", 
                 seperate_write_reward=False) -> None:
        """
        Initialize the GRPOTurnRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
            seperate_write_reward: Whether to separate write reward from answer reward. When True, uses answer_boundaries
                for turn splitting in GRPO_TURN.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.seperate_write_reward = seperate_write_reward

    def __call__(self, data: DataProto, return_dict=False):
        """GRPO_TURN adapted reward computation.
        
        This method computes rewards for each response while automatically generating
        turn_split_indices needed for GRPO_TURN advantage estimator.
        
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

        turn_split_indices = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            turn_split_index = None

            if self.seperate_write_reward:
                answer_boundaries = data.meta_info.get('answer_boundaries', {})
                answer_boundary = answer_boundaries.get(i, None)
                write_rewards = data.meta_info.get('write_rewards', {})
                write_reward = write_rewards.get(i, 0.0)

                if answer_boundary is not None and answer_boundary < valid_response_length:
                    reward_tensor[i, answer_boundary] = reward
                    reward_tensor[i, valid_response_length - 1] = write_reward
                    turn_split_index = answer_boundary
                    data.meta_info['answer_boundaries'][i] = answer_boundary
                else:
                    reward_tensor[i, valid_response_length - 1] = reward
                    turn_split_index = max(0, valid_response_length.item() // 2 - 1)
                    data.meta_info['answer_boundaries'][i] = valid_response_length - 1
            else:
                reward_tensor[i, valid_response_length - 1] = reward
                turn_split_index = max(0, valid_response_length.item() // 2 - 1)

            if turn_split_index is not None:
                turn_split_indices.append(turn_split_index)
            else:
                turn_split_indices.append(max(0, valid_response_length.item() // 2 - 1))

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
                
                if self.seperate_write_reward:
                    print(f"[turn_split_index]", turn_split_indices[-1])

        if 'turn_split_indices' not in data.non_tensor_batch:
            data.non_tensor_batch['turn_split_indices'] = np.array(turn_split_indices, dtype=np.int32)
            print(f"[GRPOTurnRewardManager] Created turn_split_indices for GRPO_TURN: {turn_split_indices[:5]}{'...' if len(turn_split_indices) > 5 else ''}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor