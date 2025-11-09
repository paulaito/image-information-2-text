setup-env:
	curl -LsSf https://astral.sh/uv/install.sh | sh

install: setup-env
	uv sync
	uv run python -m scripts.download_data
	
start-ollama:
	docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama  

stop-ollama:
	docker rm ollama

pull-ollama-models:
	docker exec -it ollama ollama pull qwen3-vl:2b 
	docker exec -it ollama ollama pull granite3.2-vision:2b