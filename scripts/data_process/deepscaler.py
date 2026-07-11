import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
import argparse
from datasets import concatenate_datasets
import random
import json

def make_prefix(dp, template_type):
    if template_type == "baseline_boxed":
        prefix = f"""Let's think step by step and output the final answer within \\boxed{{}}.
Question: """
        return prefix + dp['problem']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/deepscaler/', 
                        help='Local directory to save processed data')
    parser.add_argument('--template_type', type=str, default='baseline_boxed', choices=["baseline_boxed"])

    args = parser.parse_args()

    all_train_datasets = []
    all_test_datasets = []

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            
            target = example.get('answer')
            
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
                    'original_question': example.get('problem', ''),
                },
                "data_source": "deepscaler"
            }
            return data
        return process_fn

    print("Loading dataset from HuggingFace...")
    file_path = "agentica-org/DeepScaleR-Preview-Dataset"
    dataset = load_dataset(file_path, split='train')
    print(f"Dataset loaded successfully!")

    dataset = dataset.shuffle(seed=42)
    split_dataset = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    print("Processing train dataset...")
    train_dataset = train_dataset.map(
        function=make_map_fn('train'), 
        with_indices=True,
        desc="Processing train data"
    )
    
    print("Processing test dataset...")
    test_dataset = test_dataset.map(
        function=make_map_fn('test'), 
        with_indices=True,
        desc="Processing test data"
    )

    local_dir = os.path.join(args.local_dir, args.template_type)
    os.makedirs(local_dir, exist_ok=True)
    print(f"Saving data to {local_dir}")

    train_path = os.path.join(local_dir, 'train.parquet')
    train_dataset.to_parquet(train_path)
    print(f"Saved train dataset to {train_path}")

    test_500_path = os.path.join(local_dir, 'test_500.parquet')
    test_dataset.to_parquet(test_500_path)
    print(f"Saved test 500 dataset to {test_500_path} (size: {len(test_dataset)})")

    train_5000_dataset = train_dataset.train_test_split(test_size=5000, seed=42)["test"]
    train_5000_path = os.path.join(local_dir, 'train_5000.parquet')
    train_5000_dataset.to_parquet(train_5000_path)
    print(f"Saved sft train 5000 dataset to {train_5000_path} (size: {len(train_5000_dataset)})")

    train_10000_dataset = train_dataset.train_test_split(test_size=10000, seed=42)["test"]
    train_10000_path = os.path.join(local_dir, 'train_10000.parquet')
    train_10000_dataset.to_parquet(train_10000_path)
    print(f"Saved sft train 10000 dataset to {train_10000_path} (size: {len(train_10000_dataset)})")

    print("\n" + "="*50)
    print("Dataset processing completed!")
    print(f"Total train samples: {len(train_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")
    print(f"Data saved to: {local_dir}")
    print("="*50)
