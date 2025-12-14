# Image Information to Text

This project leverages VLMs to extract image information, namely from diagrams and charts.

How to install dependencies:
```make install```

How to start ollama:
```make start-ollama```

How to run with CPU device:
```uv run python entrypoint.py --config_path config/cpu.yaml```

How to run with GPU device:
```uv run python entrypoint.py --config_path config/gpu.yaml```

:warning: This is a WIP.

- Qwen models: require [license](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE) for commercial use.

LLava model: [Apache License](https://github.com/haotian-liu/LLaVA/blob/main/LICENSE)
- [Paper](http://arxiv.org/abs/2304.08485)

LLava-LLama model: [Apache License](https://github.com/InternLM/xtuner/blob/main/LICENSE)
- LLava[Paper](http://arxiv.org/abs/2304.08485)
- LLama [paper](http://arxiv.org/abs/2302.13971)

Granite vision model (IBM): [Apache License](https://github.com/InternLM/xtuner/blob/main/LICENSE)
- [Paper](http://arxiv.org/abs/2502.09927)

Moondream model: [Apache License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
- [Blog post](https://www.analyticsvidhya.com/blog/2024/03/introducing-moondream2-a-tiny-vision-language-model/)