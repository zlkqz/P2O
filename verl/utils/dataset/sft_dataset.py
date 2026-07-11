"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: str | ListConfig, tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get("use_shm", False)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, tuple | list) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, tuple | list) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses = self.responses.tolist()

    def __len__(self):
        return len(self.prompts)
    
    def _find_tag_positions(self, text, start_tag, end_tag):
        """
        Find the positions of content between start_tag and end_tag in text.
        Returns a list of tuples (start_pos, end_pos) for each occurrence.
        Note: The returned positions include the tags themselves.
        """
        positions = []
        search_start = 0
        
        while True:
            start_pos = text.find(start_tag, search_start)
            if start_pos == -1:
                break
            
            end_pos = text.find(end_tag, start_pos + len(start_tag))
            if end_pos == -1:
                break
            
            tag_start = start_pos
            tag_end = end_pos + len(end_tag)
            positions.append((tag_start, tag_end))
            
            search_start = tag_end
        
        return positions

    def _get_token_positions_for_text_range(self, full_text, text_start, text_end, tokenizer_output):
        """
        Map text character positions to token positions.
        """
        char_to_token = tokenizer_output.char_to_token
        
        start_token = None
        end_token = None
        
        for char_pos in range(text_start, len(full_text)):
            token_idx = char_to_token(char_pos)
            if token_idx is not None:
                start_token = token_idx
                break
        
        for char_pos in range(text_end - 1, text_start - 1, -1):
            token_idx = char_to_token(char_pos)
            if token_idx is not None:
                end_token = token_idx + 1
                break
        
        return start_token, end_token

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        prompt_chat = [{"role": "user", "content": prompt}]

        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token

        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0

        information_positions = self._find_tag_positions(response_chat_str, '<information>', '</information>')
        golden_answer_positions = self._find_tag_positions(response_chat_str, '<golden_answer>', '</golden_answer>')

        all_masked_positions = information_positions + golden_answer_positions
        
        if all_masked_positions:
            response_encoding = tokenizer(response_chat_str, add_special_tokens=False, return_offsets_mapping=True)
            offset_mapping = response_encoding['offset_mapping']
            
            for text_start, text_end in all_masked_positions:
                token_start = None
                token_end = None
                
                for token_idx, (char_start, char_end) in enumerate(offset_mapping):
                    if char_start == char_end:
                        continue
                    
                    if token_start is None and char_end > text_start:
                        token_start = token_idx
                    
                    if char_start < text_end:
                        token_end = token_idx + 1
                
                if token_start is not None and token_end is not None:
                    actual_start = prompt_length + token_start
                    actual_end = prompt_length + token_end
                    actual_start = min(actual_start, loss_mask.size(0))
                    actual_end = min(actual_end, loss_mask.size(0))
                    if actual_start < actual_end:
                        mask_start = max(prompt_length - 1, actual_start - 1)
                        mask_end = actual_end - 1
                        if mask_start < mask_end and mask_end <= loss_mask.size(0):
                            loss_mask[mask_start:mask_end] = 0
                            

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
