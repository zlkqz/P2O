from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("grouped_valid")
class GroupedValidRewardManager:
    """
    通过data[i].non_tensor_batch.extra_info中的group_id将不同的sample划分组
    然后每个划分好的组统一计算answer score
    一个组的所有样本需要answer score都为1, reward才为1; 反之为0 
    ** 聚合的只有answer score, format score是不动的 **
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

        self.sample_group_ids = []
        self.sample_end_token_indices = []
        self.sample_answer_score = []

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

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
                answer_score = score["answer_score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                answer_score = score

            group_id = extra_info.get("group_id", None)
            assert group_id is not None, "[Error]: Haven't set group_id"
            self.sample_group_ids.append(group_id)
            self.sample_end_token_indices.append(valid_response_length - 1)
            self.sample_answer_score.append(answer_score)

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

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def get_grouped_scores(self):
        answer_scores = []
        group_all_one = {}
        for idx, group_id in enumerate(self.sample_group_ids):
            is_one = 1 if self.sample_answer_score[idx] == 1 else 0
            if group_id not in group_all_one:
                group_all_one[group_id] = is_one == 1
            else:
                group_all_one[group_id] = group_all_one[group_id] and (is_one == 1)

        for i in range(len(self.sample_group_ids)):
            group_id = self.sample_group_ids[i]
            group_answer_score = 1.0 if group_all_one.get(group_id, False) else 0.0
            answer_scores.append(group_answer_score)

        self.sample_group_ids = []
        self.sample_end_token_indices = []
        self.sample_answer_score = []

        return answer_scores

