from typing import Type
from config.default import MazeAIConfig

AVAILABLE_CONFIGS: dict[str, Type[MazeAIConfig]] = {
    "default": MazeAIConfig
}
