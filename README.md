# Image Information to Text

This project leverages VLMs to extract image information, namely from diagrams and charts.

How to install dependencies:
```make install```

How to start ollama:
```make start-ollama```

How to run with CPU device:
```uv run python entrypoint.py --config_path config/cpu.yaml```

How to run with GPU device:
```uv run python entrypoint.py --config_path config/cpu.yaml```

:warning: This is a WIP.

Qwen models: require [license](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE) for commercial use.
LLava-LLama model: [Apache License](https://github.com/InternLM/xtuner/blob/main/LICENSE)
Granite vision model (IBM): [Apache License](https://github.com/InternLM/xtuner/blob/main/LICENSE)
LLava model: [Apache License](https://github.com/haotian-liu/LLaVA/blob/main/LICENSE)
Moondream model: [Apache License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)