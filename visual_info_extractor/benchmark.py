from visual_info_extractor.io import DataIO
from visual_info_extractor.inferencer import VLMInferencer
from visual_info_extractor.logger import logging
from visual_info_extractor.config import AppConfig
from visual_info_extractor.ollama.healthcheck import OllamaHealthChecker
from visual_info_extractor.ollama.client import OllamaClient

import subprocess

class Benchmark:
    def __init__(self, config_path: str, download: bool = False):
        self.io = DataIO()
        self.config = AppConfig.from_yaml(config_path)
        
        self.hf_datasets = self.config.hf_datasets
        self.download = download

        self.set_client()
        self.model_names = [model.name for model in self.config.models]
        self.client.pull_models(self.model_names)
    
    def download_data(self):
        for dataset in self.hf_datasets:
            self.io.download(
                path=dataset.path,
                write_to_path=dataset.save_to,
                dataset_name=dataset.name,
                split=dataset.split,
                num_samples=dataset.sample,
            )

    def set_client(self):
        connection_status = OllamaHealthChecker(host=self.config.ollama_host).check_connection()
        if connection_status:
            self.client = OllamaClient(host=self.config.ollama_host)

    def set_inferencer(self, model, dataset) -> VLMInferencer:
        prompt = model.prompt if model.prompt else self.config.default_prompt
        name = model.name
        save_to = dataset.save_to
        version = self.config.version
        dataset_dir = f"{dataset.save_to}/{dataset.name}"
        results_dir = f"{dataset.save_to}/results/{self.config.version}"

        if not prompt or not name or not save_to or not version:
            raise LookupError("Missing model parameter")
        
        logging.info(f"""
        Creating Inferencer with following parameters:
            version: {version}
            model name: {name}
            model prompt: {prompt}
            dataset directory: {dataset_dir}
            results directory: {results_dir}""")
        
        return VLMInferencer(
            model_name=name,
            datasets_dir=dataset_dir,
            results_dir=results_dir,
            ollama_client=self.client,
            io=self.io,
            prompt=prompt
        )

    def run(self) -> None:
        if self.download:
            self.download_data()
        
        for model in self.config.models:
            for dataset in self.hf_datasets:
                inferencer = self.set_inferencer(model,
                                                 dataset=dataset)
                inferencer.set_prompt(model.prompt)
                inferencer.inference()