# VLMs Benchmark

This project leverages VLMs to extract image information, namely from diagrams and charts. It also applies LLM-as-a-Judge to validate the VLMs benchmarking, allowing to extract metrics like accuracy from long-form responses.

## How to Run

How to install dependencies:
```make install```

How to start ollama:
```make start-ollama```

How to run benchmarking:
```uv run python entrypoint.py --config_path config/gpu.yaml```

If this is your first time running the project, make sure you call it with ```--download``` parameter to download the data. Example for CPU:
```uv run python entrypoint.py --config_path config/cpu.yaml --download```

Another option to download the data without applying the inferencer is by using the following Make command:
```make download-data```

How to run evaluator:
```uv run python entrypoint.py --config_path config/evaluator.yaml --is_eval true```

## Datasets 
Hugging Face subsets from [the_cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) dataset:
- [diagram_image_to_text](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron/viewer/diagram_image_to_text)
- [chart2text](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron/viewer/chart2text)

## VLMs Considered
Qwen models: require [license](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE) for commercial use.

LLava model: [Apache License](https://github.com/haotian-liu/LLaVA/blob/main/LICENSE)
- [Paper](http://arxiv.org/abs/2304.08485)

LLava-LLama model: [Apache License](https://github.com/InternLM/xtuner/blob/main/LICENSE)
- LLava [paper](http://arxiv.org/abs/2304.08485)
- LLama [paper](http://arxiv.org/abs/2302.13971)

Granite vision model (IBM): [Apache License](https://github.com/InternLM/xtuner/blob/main/LICENSE)
- [Paper](http://arxiv.org/abs/2502.09927)

Moondream model: [Apache License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
- [Publication](https://www.analyticsvidhya.com/blog/2024/03/introducing-moondream2-a-tiny-vision-language-model/)