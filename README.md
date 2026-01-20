# Image Information to Text

This project leverages VLMs to extract image information, namely from diagrams and charts. It also applies LLM-as-a-Judge to validate the VLMs benchmarking, allowing to extract metrics like accuracy from long-form responses.

How to install dependencies:
```make install```

How to start ollama:
```make start-ollama```

How to run benchmarking:
```uv run python entrypoint.py --config_path config/gpu.yaml```

How to run evaluator:
```uv run python entrypoint.py --config_path config/evaluator.yaml --is_eval true```