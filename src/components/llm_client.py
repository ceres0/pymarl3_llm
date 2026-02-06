import os
from typing import Optional, Dict, Any, List
from anthropic import Anthropic
from dotenv import load_dotenv


class LLMClient:
    """
    LLM Client for interacting with Large Language Models
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ):
        """
        Initialize LLM Client

        Args:
            api_key: LLM API key. If None, will load from environment variable
            base_url: Optional custom base URL for API endpoints (for proxies or self-hosted models)
            model: Model name to use. If None, will load from environment variable
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
        """
        # Load environment variables
        load_dotenv()

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        if not self.api_key or self.api_key == "your_llm_api_key_here":
            raise ValueError(
                "LLM_API_KEY not found. Please set it in .env file or pass as parameter."
            )

        # Get base_url from parameter or environment
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        if not self.base_url or self.base_url == "your_llm_base_url_here":
            raise ValueError(
                "LLM_BASE_URL not found. Please set it in .env file or pass as parameter."
            )

        # Initialize llm client with base_url
        self.client = Anthropic(api_key=self.api_key, base_url=self.base_url)

        # Configuration
        self.model = model or os.getenv("LLM_MODEL")
        if not self.model or self.model == "your_llm_model_here":
            raise ValueError(
                "LLM_MODEL not found. Please set it in .env file or pass as parameter"
            )

        self.max_tokens = max_tokens
        self.temperature = temperature

        self._initialized = False

    def verify_connection(self) -> bool:
        """
        Verify the connection to the LLM API

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Send a simple test message
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[
                    {
                        "role": "user",
                        "content": "Reply with 'OK' if you can read this."
                    }
                ]
            )

            self._initialized = True
            return True
        except Exception as e:
            print(f"LLM API Connection Error: {str(e)}")
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text from the LLM

        Args:
            prompt: User prompt/query
            system_prompt: Optional system prompt to set behavior
            max_tokens: Override default max_tokens
            temperature: Override default temperature
            **kwargs: Additional parameters to pass to the API

        Returns:
            str: Generated text response
        """
        if not self._initialized:
            raise RuntimeError(
                "LLM client not initialized. Call verify_connection() first.")

        messages = [{"role": "user", "content": prompt}]

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "messages": messages
        }

        # Add system prompt if provided
        if system_prompt:
            api_params["system"] = system_prompt

        # Add any additional parameters
        api_params.update(kwargs)

        try:
            response = self.client.messages.create(**api_params)
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {str(e)}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Multi-turn conversation with the LLM

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            system_prompt: Optional system prompt
            max_tokens: Override default max_tokens
            temperature: Override default temperature
            **kwargs: Additional parameters

        Returns:
            str: Generated response
        """
        if not self._initialized:
            raise RuntimeError(
                "LLM client not initialized. Call verify_connection() first.")

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "messages": messages
        }

        # Add system prompt if provided
        if system_prompt:
            api_params["system"] = system_prompt

        # Add any additional parameters
        api_params.update(kwargs)

        try:
            response = self.client.messages.create(**api_params)
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {str(e)}")

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration

        Returns:
            dict: Current configuration parameters
        """
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "base_url": self.base_url,
            "initialized": self._initialized
        }


def create_llm_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to create and initialize an LLM client

    Args:
        api_key: Optional API key
        model: Optional model name
        **kwargs: Additional configuration parameters

    Returns:
        LLMClient: Initialized LLM client
    """
    client = LLMClient(api_key=api_key, model=model, **kwargs)

    if not client.verify_connection():
        raise RuntimeError(
            "Failed to connect to LLM API. Please check your API key and network connection.")

    return client


if __name__ == "__main__":
    print("")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  LLM Client Test".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")

    # Run tests
    client = LLMClient()

    print("Conecting to LLM API...")
    if client.verify_connection():
        print("Connection successful!")
    else:
        print("Connection failed!")

    print()
