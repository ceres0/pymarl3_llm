from .run import run as default_run
from .run_with_llm import run as run_with_llm

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["run_with_llm"] = run_with_llm
