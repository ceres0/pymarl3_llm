import os
import yaml
from typing import Any, Optional

from .llm_client import LLMClient


class LLMGenerator:
    PROMPT_KEYS = {
        "first_user_prompt_with_example",
        "improvement_prompt",
    }

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        output_dir: str = "generated",
        prompts_name: str = "default",
    ) -> None:
        # Initialize LLM client
        self.llm_client = llm_client or LLMClient()
        if not self.llm_client._initialized:
            if not self.llm_client.verify_connection():
                raise RuntimeError("Failed to connect to LLM Client")

        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # load prompts
        self.prompts = {}

        with open(os.path.join(os.path.dirname(__file__), "config", "prompts", "{}.yaml".format(prompts_name)),
                  "r") as f:
            try:
                self.prompts = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(prompts_name, exc)
