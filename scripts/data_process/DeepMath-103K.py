from typing import Any


import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
import argparse
from datasets import concatenate_datasets
import random
import json
import re
from collections import Counter, defaultdict
import math


def classify_answer_type(answer):
    if answer is None or answer == "":
        return "empty"

    answer_str = str(answer).strip()
    answer_lower = answer_str.lower()

    if answer_lower in ['yes', 'no', 'true', 'false']:
        return "yes_no"

    if re.match(r'^[A-Za-z]$', answer_str) or re.match(r'^\([A-Za-z]\)$', answer_str):
        return "multiple_choice"

    if re.match(r'^-?\d+$', answer_str):
        return "numeric"
    if re.match(r'^-?\d+\.\d+$', answer_str):
        return "numeric"
    if re.match(r'^-?\d+/\d+$', answer_str):
        return "numeric"

    if re.match(r'^-?\d+\.?\d*%$', answer_str):
        return "percentage"

    if re.match(r'^-?\d+\.?\d*\s*[a-zA-Z]+$', answer_str):
        return "numeric_with_unit"

    if any(char in answer_str for char in ['√', 'π', '∞', '∑', '∫']):
        return "mathematical_symbol"

    if re.match(r'^[\[\(]-?.*,.*[\]\)]$', answer_str):
        return "interval_set"

    if re.match(r'^\(-?\d+\.?\d*,-?\d+\.?\d*(,-?\d+\.?\d*)?\)$', answer_str):
        return "coordinate_vector"

    if len(answer_str.split()) > 5:
        return "text_answer"

    return "other"


def balance_dataset_by_answer_type(ds, target_yes_no_ratio=0.05, smoothing_power=0.5, seed=42):
    random.seed(seed)

    difficulty_groups = defaultdict(list)
    for idx, example in enumerate(ds):
        difficulty = example.get('difficulty', 0)
        answer = example.get('final_answer', '')
        answer_type = classify_answer_type(answer)
        difficulty_groups[difficulty].append({
            'idx': idx,
            'answer_type': answer_type,
            'example': example
        })

    balanced_indices = []

    print("\n" + "="*80)
    print("Balancing dataset by answer type for each difficulty level...")
    print("="*80)

    for difficulty in sorted(difficulty_groups.keys()):
        examples = difficulty_groups[difficulty]

        type_counter = Counter([ex['answer_type'] for ex in examples])

        examples = [ex for ex in examples if ex['answer_type'] != 'empty']
        type_counter.pop('empty', None)

        if len(examples) == 0:
            continue

        total_count = len(examples)

        print(f"\nDifficulty {difficulty}: {total_count} samples")
        print(f"  Original distribution: {dict(type_counter)}")

        type_to_examples = defaultdict(list)
        for ex in examples:
            type_to_examples[ex['answer_type']].append(ex)

        sampling_counts = {}

        other_types = {k: v for k, v in type_counter.items() if k != 'yes_no'}
        if other_types:
            other_weights = {}
            for answer_type, count in other_types.items():
                other_weights[answer_type] = count ** smoothing_power

            total_other_weight = sum(other_weights.values())
            target_other_total = int(total_count * (1 - target_yes_no_ratio))

            for answer_type, weight in other_weights.items():
                sampling_counts[answer_type] = int((weight / total_other_weight) * target_other_total)

        if 'yes_no' in type_counter:
            other_total = sum(sampling_counts.values())
            max_yes_no = int(other_total * target_yes_no_ratio / (1 - target_yes_no_ratio))
            sampling_counts['yes_no'] = min(type_counter['yes_no'], max_yes_no)

        for answer_type, target_count in sampling_counts.items():
            available = type_to_examples[answer_type]
            if target_count >= len(available):
                sampled = available
            else:
                sampled = random.sample(available, target_count)

            for ex in sampled:
                balanced_indices.append(ex['idx'])

        balanced_total = sum(sampling_counts.values())
        balanced_type_counter = Counter()
        for answer_type, count in sampling_counts.items():
            if count > 0:
                balanced_type_counter[answer_type] = count

        print(f"  Balanced distribution: {dict(balanced_type_counter)}")
        print(f"  Balanced total: {balanced_total} (reduced from {total_count})")
        yes_no_ratio = balanced_type_counter.get('yes_no', 0) / balanced_total * 100 if balanced_total > 0 else 0
        print(f"  Yes/No ratio: {yes_no_ratio:.2f}%")

    random.shuffle(balanced_indices)
    balanced_ds = ds.select(balanced_indices)

    print("\n" + "="*80)
    print(f"Balancing completed! Original: {len(ds)}, Balanced: {len(balanced_ds)}")

    all_types = [classify_answer_type(ex.get('final_answer', '')) for ex in balanced_ds]
    overall_counter = Counter(all_types)
    print(f"Overall balanced distribution:")
    for answer_type, count in overall_counter.most_common():
        ratio = count / len(all_types) * 100
        print(f"  {answer_type}: {count} ({ratio:.2f}%)")

    yes_no_count = overall_counter.get('yes_no', 0)
    total_count = len(all_types)
    yes_no_ratio = yes_no_count / total_count if total_count > 0 else 0

    if yes_no_ratio > target_yes_no_ratio:
        print(f"\nWarning: Overall yes_no ratio ({yes_no_ratio*100:.2f}%) exceeds target ({target_yes_no_ratio*100:.2f}%).")
        print(f"Performing global adjustment...")

        max_yes_no_global = int(total_count * target_yes_no_ratio)

        yes_no_indices = []
        other_indices = []
        for idx, ex in enumerate(balanced_ds):
            answer_type = classify_answer_type(ex.get('final_answer', ''))
            if answer_type == 'yes_no':
                yes_no_indices.append(idx)
            else:
                other_indices.append(idx)

        if len(yes_no_indices) > max_yes_no_global:
            yes_no_keep_indices = random.sample(yes_no_indices, max_yes_no_global)
        else:
            yes_no_keep_indices = yes_no_indices

        final_indices = other_indices + yes_no_keep_indices
        random.shuffle(final_indices)

        balanced_ds = balanced_ds.select(final_indices)

        all_types = [classify_answer_type(ex.get('final_answer', '')) for ex in balanced_ds]
        overall_counter = Counter(all_types)
        print(f"\nAfter global adjustment:")
        print(f"  Total samples: {len(balanced_ds)}")
        for answer_type, count in overall_counter.most_common():
            ratio = count / len(all_types) * 100
            print(f"  {answer_type}: {count} ({ratio:.2f}%)")

    print("="*80 + "\n")

    return balanced_ds


def make_prefix(dp, template_type):
    if template_type == "baseline_boxed":
        prefix = f"""Let's think step by step and output the final answer within \\boxed{{}}.
Question: """
        return prefix + dp['question']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/deepmath/', 
                        help='Local directory to save processed data')
    parser.add_argument('--template_type', type=str, default='baseline_boxed', choices=["baseline_boxed"])

    args = parser.parse_args()

    all_train_datasets = []
    all_test_datasets = []

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            
            target = example.get('final_answer')
            
            solution = {
                "target": target,
            }
            
            data = {
                "id": f"{split}_{idx}",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "general_reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'original_question': example.get('question', ''),
                    "difficulty": example.get("difficulty", ""),
                    "topic": example.get("topic", ""),
                },
                "data_source": "deepmath"
            }
            return data
        return process_fn

    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("zwhe99/DeepMath-103K", trust_remote_code=True)
    print(f"Dataset loaded successfully!")

    base_train_ds = dataset["train"]

    def split_and_save(ds, name):
        actual_test_size = min(len(ds), 500)
        if actual_test_size < len(ds):
            split_dataset = ds.train_test_split(test_size=actual_test_size, seed=42)
            train_ds = split_dataset['train']
            test_ds = split_dataset['test']
        else:
            train_ds = ds.select([])
            test_ds = ds

        print(f"Processing train dataset for {name}...")
        train_ds = train_ds.map(
            function=make_map_fn('train'),
            with_indices=True,
            desc=f"Processing train data ({name})"
        )

        print(f"Processing test dataset for {name}...")
        test_ds = test_ds.map(
            function=make_map_fn('test'),
            with_indices=True,
            desc=f"Processing test data ({name})"
        )

        local_dir = os.path.join(args.local_dir, args.template_type, name)
        os.makedirs(local_dir, exist_ok=True)
        print(f"Saving data to {local_dir}")

        actual_test_size = len(test_ds)
        test_path = os.path.join(local_dir, f'test_{actual_test_size}.parquet')
        test_ds.to_parquet(test_path)
        print(f"Saved test dataset to {test_path} (size: {actual_test_size})")

        if len(train_ds) == 0:
            print("Warning: Train dataset is empty after test split, skipping train dataset creation.")
        else:
            train_path = os.path.join(local_dir, 'train.parquet')
            train_ds.to_parquet(train_path)
            print(f"Saved train dataset to {train_path}")

            target_train_5000_size = min(len(train_ds), 5000)
            if target_train_5000_size < len(train_ds):
                train_5000_ds = train_ds.train_test_split(test_size=target_train_5000_size, seed=42)["test"]
            else:
                train_5000_ds = train_ds
            train_5000_path = os.path.join(local_dir, f'train_{target_train_5000_size}.parquet')
            train_5000_ds.to_parquet(train_5000_path)
            print(f"Saved sft train subset to {train_5000_path} (size: {len(train_5000_ds)})")

            target_train_10000_size = min(len(train_ds), 10000)
            if target_train_10000_size < len(train_ds):
                train_10000_ds = train_ds.train_test_split(test_size=target_train_10000_size, seed=42)["test"]
            else:
                train_10000_ds = train_ds
            train_10000_path = os.path.join(local_dir, f'train_{target_train_10000_size}.parquet')
            train_10000_ds.to_parquet(train_10000_path)
            print(f"Saved sft train subset to {train_10000_path} (size: {len(train_10000_ds)})")

        print("\n" + "="*50)
        print(f"{name} dataset processing completed!")
        print(f"Total train samples: {len(train_ds)}")
        print(f"Total test samples: {len(test_ds)}")
        print(f"Data saved to: {local_dir}")
        print("="*50)

        print(f"\n{'='*80}")
        print(f"Creating balanced dataset for {name}...")
        print(f"{'='*80}")

        balanced_full_ds = balance_dataset_by_answer_type(ds, target_yes_no_ratio=0.05, smoothing_power=0.5, seed=42)

        actual_test_size_balanced = min(len(balanced_full_ds), 500)
        if actual_test_size_balanced < len(balanced_full_ds):
            balanced_split = balanced_full_ds.train_test_split(test_size=actual_test_size_balanced, seed=42)
            balanced_train_ds = balanced_split['train']
            balanced_test_ds = balanced_split['test']
        else:
            balanced_train_ds = balanced_full_ds.select([])
            balanced_test_ds = balanced_full_ds

        print(f"Processing balanced train dataset for {name}...")
        balanced_train_ds = balanced_train_ds.map(
            function=make_map_fn('train'),
            with_indices=True,
            desc=f"Processing balanced train data ({name})"
        )

        print(f"Processing balanced test dataset for {name}...")
        balanced_test_ds = balanced_test_ds.map(
            function=make_map_fn('test'),
            with_indices=True,
            desc=f"Processing balanced test data ({name})"
        )

        balanced_dir = os.path.join(args.local_dir, args.template_type, name, "balanced")
        os.makedirs(balanced_dir, exist_ok=True)
        print(f"Saving balanced data to {balanced_dir}")

        actual_test_size_balanced = len(balanced_test_ds)
        balanced_test_path = os.path.join(balanced_dir, f'test_{actual_test_size_balanced}.parquet')
        balanced_test_ds.to_parquet(balanced_test_path)
        print(f"Saved balanced test dataset to {balanced_test_path} (size: {actual_test_size_balanced})")

        if len(balanced_train_ds) > 0:
            balanced_train_path = os.path.join(balanced_dir, 'train.parquet')
            balanced_train_ds.to_parquet(balanced_train_path)
            print(f"Saved balanced train dataset to {balanced_train_path}")

            target_train_5000_size = min(len(balanced_train_ds), 5000)
            if target_train_5000_size < len(balanced_train_ds):
                balanced_train_5000_ds = balanced_train_ds.train_test_split(test_size=target_train_5000_size, seed=42)["test"]
            else:
                balanced_train_5000_ds = balanced_train_ds
            balanced_train_5000_path = os.path.join(balanced_dir, f'train_{target_train_5000_size}.parquet')
            balanced_train_5000_ds.to_parquet(balanced_train_5000_path)
            print(f"Saved balanced sft train subset to {balanced_train_5000_path} (size: {len(balanced_train_5000_ds)})")

            target_train_10000_size = min(len(balanced_train_ds), 10000)
            if target_train_10000_size < len(balanced_train_ds):
                balanced_train_10000_ds = balanced_train_ds.train_test_split(test_size=target_train_10000_size, seed=42)["test"]
            else:
                balanced_train_10000_ds = balanced_train_ds
            balanced_train_10000_path = os.path.join(balanced_dir, f'train_{target_train_10000_size}.parquet')
            balanced_train_10000_ds.to_parquet(balanced_train_10000_path)
            print(f"Saved balanced sft train subset to {balanced_train_10000_path} (size: {len(balanced_train_10000_ds)})")
        else:
            print("Warning: Balanced train dataset is empty, skipping.")

        print("\n" + "="*50)
        print(f"{name} balanced dataset processing completed!")
        print(f"Balanced train samples: {len(balanced_train_ds)}")
        print(f"Balanced test samples: {len(balanced_test_ds)}")
        print(f"Balanced data saved to: {balanced_dir}")
        print("="*50 + "\n")

    split_and_save(base_train_ds, "full")

    hard_ds = base_train_ds.filter(lambda x: x.get("difficulty", 0) >= 5)
    split_and_save(hard_ds, "hard_le_5")

    very_hard_ds = base_train_ds.filter(lambda x: x.get("difficulty", 0) >= 6)
    split_and_save(very_hard_ds, "very_hard_le_6")

    very_hard_ds = base_train_ds.filter(lambda x: x.get("difficulty", 0) >= 7)
    split_and_save(very_hard_ds, "very_hard_le_7")

    very_hard_ds = base_train_ds.filter(lambda x: x.get("difficulty", 0) >= 8)
    split_and_save(very_hard_ds, "very_hard_le_8")

    very_hard_ds = base_train_ds.filter(lambda x: x.get("difficulty", 0) >= 9)
    split_and_save(very_hard_ds, "very_hard_le_9")
