setup-env:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="$HOME/.local/bin:$PATH"

install: setup-env
	uv sync
	download-data

download-data: 
	uv run python -m scripts.download_data

start-ollama:
	docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama  

stop-ollama:
	docker rm ollama

pull-ollama-models:
	docker exec -it ollama ollama pull qwen2.5vl:3b  
	docker exec -it ollama ollama pull qwen3-vl:2b 
	docker exec -it ollama ollama pull granite3.2-vision:2b
	docker exec -it ollama ollama pull llava:7b
	docker exec -it ollama ollama pull llava-llama3:8b
	docker exec -it ollama ollama pull moondream:1.8b