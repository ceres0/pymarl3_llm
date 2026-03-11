import builtins
import logging
import os
import re
import yaml
from typing import Any, Optional

from .llm_client import LLMClient

_logger = logging.getLogger(__name__)


class LLMGenerator:
    PROMPT_KEYS = {
        "first_user_prompt_with_example",
        "improvement_prompt",
    }

    #: Directory that holds all prompt YAML files.
    _PROMPTS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "prompts"
    )

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        output_dir: str = "generated",
        prompts_name: str = "default",
        env_config_name: Optional[str] = None,
    ) -> None:
        """
        Args:
            llm_client: Pre-built :class:`LLMClient`.  A new one is created when omitted.
            output_dir: Directory where generated files are saved.
            prompts_name: Name of the general prompt YAML (``config/prompts/<name>.yaml``).
            env_config_name: Name of an *optional* environment-specific prompt YAML
                (``config/prompts/<name>.yaml``) that supplies ``env_description``,
                ``task_description_template``, and ``original_reward_function``.
                When provided these values take precedence over anything passed
                directly to :meth:`generate_reward_function`.
        """
        # Initialize LLM client
        self.llm_client = llm_client or LLMClient()
        if not self.llm_client._initialized:
            if not self.llm_client.verify_connection():
                raise RuntimeError("Failed to connect to LLM Client")

        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # load general prompts — live at src/config/prompts/<prompts_name>.yaml
        self.prompts = self._load_yaml(prompts_name)

        # load optional env-specific config
        self.env_config: dict = {}
        if env_config_name:
            self.env_config = self._load_yaml(env_config_name)

    def _load_yaml(self, name: str) -> dict:
        """Load a YAML file from the prompts directory by name (without extension)."""
        path = os.path.join(self._PROMPTS_DIR, "{}.yaml".format(name))
        with open(path, "r", encoding="utf-8") as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                raise ValueError(
                    "{}.yaml parse error: {}".format(name, exc)) from exc

    # ------------------------------------------------------------------
    # Reward-function generation
    # ------------------------------------------------------------------

    def generate_reward_function(
        self,
        env_description: Optional[str] = None,
        task_description: Optional[str] = None,
        example_reward: Optional[str] = None,
        agents_obs_spaces: str = "",
        max_retries: int = 3,
        **task_template_kwargs: Any,
    ) -> str:
        """Generate a reward function via the LLM.

        Values that are not passed explicitly are looked up in ``self.env_config``
        (populated when *env_config_name* is given to :meth:`__init__`).

        Args:
            env_description: Human-readable description of the environment.
                Falls back to ``env_config['env_description']``.
            task_description: Description of what the agents are trying to achieve.
                Falls back to ``env_config['task_description_template']`` formatted
                with ``**task_template_kwargs``.
            example_reward: Optional one-shot reference implementation.
                Falls back to ``env_config['original_reward_function']``.
            agents_obs_spaces: Optional observation / action-space description.
            max_retries: Maximum generation attempts. Each failed validation
                discards the result and calls the LLM again.  Raises
                :class:`RuntimeError` if all attempts are exhausted.
            **task_template_kwargs: Runtime values used to format the
                ``task_description_template`` (e.g. ``map_name``, ``n_agents``,
                ``n_enemies``).

        Returns:
            A Python source-code string containing the ``reward_battle`` function.
        """
        # --- resolve values: explicit arg > env_config > empty string ---
        if env_description is None:
            env_description = self.env_config.get("env_description", "") or ""

        if task_description is None:
            tmpl = self.env_config.get("task_description_template", "") or ""
            task_description = tmpl.format(
                **task_template_kwargs) if task_template_kwargs else tmpl

        if example_reward is None:
            example_reward = self.env_config.get(
                "original_reward_function") or None
        intro = self.prompts.get("intro", "")
        env_tmpl = self.prompts.get(
            "environment_description",
            "Environment: {environment_description}\n{agents_observation_action_spaces}",
        )
        task_tmpl = self.prompts.get(
            "task_description",
            "Task: {task_description}",
        )
        rf_format = self.prompts.get("reward_function_format", "")

        # -- system prompt --
        env_section = (
            env_tmpl
            .replace("{environment_description}", env_description)
            .replace("{agents_observation_action_spaces}", agents_obs_spaces)
        )
        task_section = task_tmpl.replace(
            "{task_description}", task_description)
        system_prompt = "\n\n".join(
            filter(None, [intro, env_section, task_section]))

        # -- user prompt --
        if example_reward:
            oneshot_tmpl = self.prompts.get(
                "one-shot_hint",
                "Here is an existing reward function for reference:\n{example_reward_function}",
            )
            oneshot = oneshot_tmpl.replace(
                "{example_reward_function}", example_reward)
            user_prompt = (
                "{oneshot}\n\n"
                "Based on the environment and task descriptions above, design an improved "
                "reward function named `reward_battle` that more effectively guides the agents "
                "towards achieving the task.\n{rf_format}"
            ).format(oneshot=oneshot, rf_format=rf_format)
        else:
            zeroshot = self.prompts.get("zero-shot_hint", "")
            user_prompt = (
                "{zeroshot}\n\n"
                "Based on the environment and task descriptions above, design a reward "
                "function named `reward_battle` that effectively guides the agents towards "
                "achieving the task.\n{rf_format}"
            ).format(zeroshot=zeroshot, rf_format=rf_format)

        # -- call LLM with retry on validation failure --
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
            code = self.parse_code_from_response(response)

            try:
                self._validate_reward_code(code)
            except Exception as exc:
                last_error = exc
                _logger.warning(
                    "[LLMGenerator] Attempt %d/%d: validation failed (%s: %s). "
                    "Discarding and retrying...",
                    attempt, max_retries, type(exc).__name__, exc,
                )
                continue

            # -- validation passed: persist and return --
            os.makedirs(self.output_dir, exist_ok=True)
            raw_path = os.path.join(self.output_dir, "reward_function_response.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(response)

            code_path = os.path.join(self.output_dir, "reward_function.py")
            with open(code_path, "w", encoding="utf-8") as f:
                f.write("# LLM-generated reward function\n\n")
                f.write(code)

            return code

        raise RuntimeError(
            "LLM failed to produce a valid reward_battle function after {} attempt(s). "
            "Last error: {}: {}".format(max_retries, type(last_error).__name__, last_error)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_reward_code(code: str) -> None:
        """Syntax check + lightweight dry-run of generated reward code.

        Steps:
        1. ``compile()`` — catches all ``SyntaxError`` / ``IndentationError``.
        2. ``exec()`` inside a lenient namespace where any unknown global name
           resolves to a :class:`unittest.mock.MagicMock`, so common patterns
           like ``import numpy as np`` or bare name references don't
           auto-fail.
        3. Call ``reward_battle(mock_env)`` — catches ``TypeError``,
           ``NameError``, ``AttributeError``, and other runtime errors that
           surface immediately.

        Raises:
            SyntaxError: invalid Python syntax.
            ValueError: ``reward_battle`` is missing or not callable.
            Exception: any exception raised when calling ``reward_battle``.
        """
        from unittest.mock import MagicMock  # stdlib — always available

        # 1. Syntax check
        compiled = compile(code, "<llm_generated>", "exec")

        # 2. Exec inside a lenient globals dict so unknown names don't raise
        class _LenientGlobals(dict):
            def __missing__(self, key: str):  # type: ignore[override]
                m = MagicMock(name=key)
                self[key] = m
                return m

        ns: dict = _LenientGlobals({"__builtins__": vars(builtins)})
        exec(compiled, ns)  # noqa: S102

        fn = ns.get("reward_battle")
        if fn is None or not callable(fn):
            raise ValueError(
                "Generated code does not define a callable 'reward_battle'."
            )

        # 3. Dry-run: call with a MagicMock acting as the environment ``self``
        fn(MagicMock())

    @staticmethod
    def parse_code_from_response(response: str) -> str:
        """Extract the first Python code block from an LLM response.

        Tries (in order):
        1. ```python ... ``` fenced block
        2. ``` ... ``` fenced block
        3. Raw ``def reward_battle`` definition until the next top-level ``def``
        4. The full response as a fallback
        """
        # 1. ```python ... ```
        match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 2. ``` ... ```
        match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 3. def reward_battle(self): ... until next non-indented def / class
        match = re.search(
            r"(def reward_battle\(self\):.*?)(?=\ndef |\nclass |\Z)",
            response,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()

        # 4. Fallback
        return response.strip()
