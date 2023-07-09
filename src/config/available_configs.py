from typing import Type

from src.config.gpu import GpuConfig
from src.config.default import MazeAIConfig

AVAILABLE_CONFIGS: dict[str, Type[MazeAIConfig]] = {
    "default": MazeAIConfig,
    "gpu": GpuConfig
}
