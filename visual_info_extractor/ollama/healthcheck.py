import requests
from visual_info_extractor.ollama.base import OllamaBaseClass

class OllamaHealthChecker(OllamaBaseClass):
    """Handles checking the reachability and status of the Ollama service."""

    def __init__(self, host: str = "http://localhost:11434"):
        """Initializes the health checker with the Ollama host URL."""
        super().__init__()

    def check_connection(self) -> bool:
        """
        Attempts to connect to the Ollama host's /api/tags endpoint 
        to verify service availability.
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=3)
            if response.status_code == 200:
                print(f"Ollama is reachable at {self.host}")
                return True
            else:
                raise ConnectionError(f"Ollama responded with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to Ollama at {self.host}. Did you start ollama server?")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Connection to Ollama at {self.host} timed out.")
        # You might also want a generic except for other request issues
        except requests.exceptions.RequestException as e:
            raise Exception(f"An unexpected request error occurred: {e}")