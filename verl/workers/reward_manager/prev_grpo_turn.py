class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0., seperate_write_reward=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.seperate_write_reward = seperate_write_reward

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        
        if 'answer_boundaries' not in data.meta_info:
            data.meta_info['answer_boundaries'] = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            answer_score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            if self.seperate_write_reward:
                answer_boundaries = data.meta_info.get('answer_boundaries', {})
                answer_boundary = answer_boundaries.get(i, None)

                write_rewards = data.meta_info.get('write_rewards', {})
                write_reward = write_rewards[i]

                if answer_boundary is not None and answer_boundary < valid_response_length:
                    reward_tensor[i, answer_boundary] = answer_score
                    reward_tensor[i, valid_response_length - 1] = write_reward
                else:
                    reward_tensor[i, valid_response_length - 1] = answer_score
                    data.meta_info['answer_boundaries'][i] = valid_response_length - 1
                    region_info = f"No boundary found, answer reward at pos {valid_response_length-1} = {answer_score:.2f}"

            else:
                reward_tensor[i, valid_response_length-1] = answer_score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

        return reward_tensor