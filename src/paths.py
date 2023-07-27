import os
from src.new_config import MazeAIConfig



# class Paths(MazeAIConfig):
#     def __init__(self, config: MazeAIConfig):
#         super().__init__(config)
#
#     def has_model(self) -> bool:
#         return len(self.output.model) > 0
#
#     def output_path(self) -> str:
#         root_path = os.path.dirname(os.path.dirname(__file__))
#         return os.path.join(root_path, self.output.dir)
#
#     def model_directory(self) -> str:
#         return os.path.join(self.output_path(), self.output.model)
#
#     def model_path(self) -> str:
#         return os.path.join(self.model_directory(), self.output.model)
#
#     def data_directory(self) -> str:
#         return os.path.join(self.output_path(), self.output.data)
#
#     def tokenizer_path(self) -> str:
#         return os.path.join(self.output_path(), self.output.tokenizer)
