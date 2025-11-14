from pydantic_settings import BaseSettings
from pydantic import BaseModel
from typing import List, Optional
import yaml

class ModelConfig(BaseModel):
    name: str
    prompt: Optional[str] = None

class HFDatasetConfig(BaseModel):
    name: str
    path: str
    split: Optional[str] = "train"
    sample: Optional[int] = 20
    save_to: Optional[str] = "data"

class AppConfig(BaseSettings):
    version: str
    ollama_host: str
    models: List[ModelConfig]
    hf_datasets: List[HFDatasetConfig]
    default_prompt: str

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)