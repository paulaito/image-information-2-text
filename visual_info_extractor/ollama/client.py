import ollama
import time
import psutil
import os
import subprocess
import json
from typing import List, Dict, Any

from visual_info_extractor.ollama.base import OllamaBaseClass

class OllamaClient(OllamaBaseClass):
    """Manages interactions with the Ollama API client."""

    def __init__(self, host: str = "http://localhost:11434"):
        """Initializes the underlying ollama client."""
        # Store the client as an instance attribute
        super().__init__()

        self.client = ollama.Client(host=host)

    def get_installed_models(self) -> List[str]:
        """Returns a list of installed models by calling `ollama list` inside the container."""
        try:
            result = subprocess.run(
                ["docker", "exec", "ollama", "ollama", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            lines = result.stdout.strip().split("\n")
            # Skip header line and extract first column (model name)
            models = [line.split()[0] for line in lines[1:] if line.strip()]
            return models
        except subprocess.CalledProcessError as e:
            print(f"Error checking installed models: {e}")
            return []

    def is_model_pulled(self, model_name: str) -> bool:
        """Validates if model is pulled."""
        installed_models = self.get_installed_models()
        return model_name in installed_models

    def pull_models(self, model_names: List[str]):
        """Pulls models in case the model is not already pulled."""
        for name in model_names:
            if name and not self.is_model_pulled(name):
                cmd = ["docker", "exec", "-it", "ollama", "ollama", "pull", name]
                print(f"Pulling model: {name}")
                subprocess.run(cmd, check=True)
            else:
                print(f"Model '{name}' is already installed. Skipping.")

    def run_chat_image(self, model: str, prompt: str, image_paths: List[str]) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
        """
        Runs a chat completion request, supporting multimodal inputs (e.g., images).

        Args:
            model: The Ollama model name (e.g., 'qwen3-vl:2b').
            messages: A list of message dictionaries for the conversation.
        
        Returns:
            response: The full response from the Ollama API.
            content: The content of the assistant's reply.
            trace: A dictionary containing trace information such as execution time and resource usage.
        """

        process = psutil.Process(os.getpid())
        start_time = time.time()
        start_cpu = process.cpu_percent(interval=None)
        start_mem = process.memory_info().rss  # in bytes

        try:
            response = self.client.chat(
                model=model, 
                messages=[
                    {'role': 'user', 
                     'content': prompt, 
                     'images': image_paths
                     }
                ])
            
            # Getting trace information
            end_time = time.time()
            end_cpu = process.cpu_percent(interval=None)
            end_mem = process.memory_info().rss

            duration = end_time - start_time
            cpu_used = end_cpu - start_cpu
            mem_used = end_mem - start_mem

            trace = {
                'duration': duration,
                'cpu_used': cpu_used,
                'mem_used': mem_used
            }

            print(f"Execution time: {duration:.2f} seconds")
            print(f"Memory used: {mem_used / (1024 ** 2):.2f} MB")
            print(f"CPU usage change: {cpu_used:.2f}%")

            return response, trace
        
        except Exception as e:
            print(f"Error during chat request with model {model}: {e}")
            raise
