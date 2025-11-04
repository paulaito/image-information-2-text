import requests
from image_information_retrieval.ollama.base import OllamaBaseClass

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
                print(f"✅ Ollama is reachable at {self.host}")
                return True
            else:
                print(f"⚠️ Ollama responded with status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"❌ Could not connect to Ollama at {self.host}")
            return False
        except requests.exceptions.Timeout:
            print(f"⏱️ Connection to Ollama at {self.host} timed out")
            return False
        # You might also want a generic except for other request issues
        except requests.exceptions.RequestException as e:
            print(f"🚨 An unexpected request error occurred: {e}")
            return False