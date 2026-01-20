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

class BaseAppConfig(BaseSettings):
    version: str
    ollama_host: str

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

class AppConfig(BaseAppConfig):
    models: List[ModelConfig]
    hf_datasets: List[HFDatasetConfig]
    default_prompt: str

class EvaluatorConfig(BaseAppConfig):
    input_dir: str
    output_dir: str
    evaluator_model: str