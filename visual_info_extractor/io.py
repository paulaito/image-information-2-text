from visual_info_extractor.logger import logging
import pandas as pd
from pathlib import Path
from datasets import load_dataset
import logging

        
class DataIO:
    """
    A class responsible for common data reading
    and writing operations.
    """
    # The base path for saving downloaded evaluation datasets
    BASE_EVAL_PATH = Path("data")

    @staticmethod
    def download(path: str, 
                 write_to_path: str, 
                 dataset_name: str, 
                 split: str = "train", 
                 num_samples: int = 20) -> None:
        """
        Downloads a sample of a multi-modal Hugging Face dataset,
        saving images as PNG and corresponding texts as TXT files locally.

        Args:
            path (str): The path or name of the dataset repository on Hugging Face.
            dataset_name (str): The specific configuration/subset name of the dataset.
            split (str): The split to download (e.g., 'train', 'validation').
            num_samples (int): The number of samples to download.
        """

        logging.info(f"Downloading {dataset_name} from {path} for split {split} and {num_samples} samples...")

        try:
            ds = load_dataset(path, dataset_name, split=split)
            
            write_to_path = Path(write_to_path)
            dataset_path = write_to_path / dataset_name
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            ds_sample = ds.shuffle(seed=42).select(range(num_samples))

            for i, example in enumerate(ds_sample):
                image = example['images'][0]
                ground_truth_text = "\n".join([f"{ann}\n" for ann in example['texts']])

                # Save image
                image_filepath = dataset_path / f"{dataset_name}_{i}.png"
                image.save(image_filepath)

                # Save text
                text_filepath = dataset_path / f"{dataset_name}_{i}.txt"
                with open(text_filepath, "w") as f:
                    f.write(ground_truth_text)
            
            logging.info(f"Successfully saved {num_samples} samples to: {dataset_path}")

        except Exception as e:
            logging.error(f"Error during dataset download: {e}")

    @staticmethod
    def read_directory(
        path: str, 
        pattern: str = "*.csv",
        is_result_dir: bool = True
        ) -> pd.DataFrame:
        """
        Reads all CSV files in a directory and returns a combined DataFrame.

        Args:
            path (str): The path to the directory containing CSV files.
            pattern (str): The glob pattern to match files. Defaults to '*.csv'.
            is_result_dir (bool): Whether the directory contains result files that need special handling. Defaults to True.
        """

        logging.info(f"Reading data from {path} with pattern {pattern}...")

        try:
            dfs = []
            for f in Path(path).glob(pattern):
                tmp = pd.read_csv(f)
                if is_result_dir:
                    tmp["model"] = f.name.split("_")[1].split(".csv")[0]
                dfs.append(tmp)

            df = pd.concat(dfs, ignore_index=True)

            return df[df.iloc[:, 0] != df.columns[0]]

        except Exception as e:
            logging.error(f"Error during dataset download: {e}")

    @staticmethod
    def write(df: pd.DataFrame, file_name: str, output_dir: str, append: bool) -> None:
        """
        Writes a list of dictionaries to a local Parquet file.

        Args:
            data (list[dict]): The data to write (e.g., list of records/results).
            file_name (str): The name of the output Parquet file (e.g., 'results.parquet').
            output_dir (Path): The directory to save the file in. Defaults to DataIO.BASE_EVAL_PATH.
        """
        # Ensure the output directory exists
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
            
        output_path = output_dir / file_name

        logging.info(f"Writing data to file: {output_path.absolute()}")

        try:
            df.to_csv(output_path, index=False, mode='a' if append else 'w',)

        except Exception as e:
            logging.error(f"Error writing Parquet file to {output_path}: {e}")