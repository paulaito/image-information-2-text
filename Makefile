setup-env:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="$$HOME/.local/bin:$$PATH"

install: setup-env
	uv sync
	make download-data

download-data:
	uv run python -m scripts.download_data

start-ollama:
	@echo "Using native Ollama (no Docker)"
	@ollama list > /dev/null

stop-ollama:
	@echo "Native Ollama runs as a system service (nothing to stop)"

pull-ollama-models:
	ollama pull qwen2.5vl:3b
	ollama pull qwen3-vl:2b
	ollama pull granite3.2-vision:2b
	ollama pull llava:7b
	ollama pull llava-llama3:8b
	ollama pull moondream:1.8b
