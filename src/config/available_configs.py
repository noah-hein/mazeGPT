from typing import Type
from src.config.default import MazeAIConfig

AVAILABLE_CONFIGS: dict[str, Type[MazeAIConfig]] = {
    "default": MazeAIConfig
}
