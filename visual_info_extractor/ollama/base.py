class OllamaBaseClass:
    """Manages interactions with the Ollama API client."""

    def __init__(self, host: str = "http://localhost:11434"):
        """Initializes the underlying ollama client."""
        self.host = host