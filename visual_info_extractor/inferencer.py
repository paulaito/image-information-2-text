from visual_info_extractor.ollama.client import OllamaClient
from visual_info_extractor.io import DataIO
from visual_info_extractor.logger import logging

import glob
import pandas as pd

class VLMInferencer:
    def __init__(self, model_name: str, prompt: str, datasets_dir: str, results_dir: str, sample: int, ollama_client: OllamaClient, io: DataIO):
        self.model_name = model_name
        self.results_dir = results_dir
        self.images_path = sorted(glob.glob(f"{datasets_dir}/**/*.png", recursive=True))[:sample]
        self.txt_paths = sorted(glob.glob(f"{datasets_dir}/**/*.txt", recursive=True))[:sample]
        self.ollama_client = ollama_client
        self.io = io
        self.results = []
        self.prompt = self.set_prompt(prompt)
    
    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

    def run(self):
        self.inference()
        self.write_results()

    def inference(self):
        num_runs = len(self.images_path)
        run = 0
        for image_path, txt_path in zip(self.images_path, self.txt_paths):
            run += 1
            with open(txt_path, "r", encoding="utf-8") as f:
                groundtruth = f.read()

            response, trace = self.ollama_client.run_chat_image(
                model=self.model_name, 
                prompt=self.prompt, image_paths=[image_path])
            
            self.results.append({
                image_path: {"response": response,
                             "trace": trace,
                             "groundtruth": groundtruth,
                             "prompt": self.prompt}
            })

            logging.info(f"Finished inference: {run}/{num_runs}")
    
    def write_results(self):
        rows = []
        for item in self.results:
            for image_path, content in item.items():
                row = {"image_path": image_path, **content}
                rows.append(row)

        df = pd.DataFrame(rows)

        self.io.write(
            df = df,
            file_name=f"results_{self.model_name}.csv",
            output_dir=self.results_dir,
            append=True
        )
