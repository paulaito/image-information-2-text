from pathlib import Path
from datasets import load_dataset
from visual_info_extractor.logger import logging

BASE_EVAL_PATH = Path("../data/evaluation")

def download_hf_sample_data(path, dataset_name, split="train", num_samples=20):
    logging.info(f"Downloading {dataset_name} from {path} for split {split} and {num_samples} samples...")
    ds = load_dataset(path, dataset_name, split=split)
    dataset_path = BASE_EVAL_PATH / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    ds_sample = ds.shuffle(seed=42).select(range(num_samples))

    for i, example in enumerate(ds_sample):
        image = example['images'][0]
        ground_truth_text = "\n".join([f"{ann}\n" for ann in example['texts']])

        image.save(dataset_path / f"{dataset_name}{i}.png")
        with open(dataset_path / f"{dataset_name}{i}.txt", "w") as f:
            f.write(ground_truth_text)

    logging.info(f"Done!")