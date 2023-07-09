![MazeGPT](media/logo_small.png)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![GitHub release](https://img.shields.io/github/release/noah-hein/mazeGPT)](https://GitHub.com/noah-hein/mazeGPT/releases/)
![Repo Size](https://img.shields.io/github/repo-size/noah-hein/mazeGPT)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/noah-hein/mazeGPT/blob/main/LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/noah-hein/mazeGPT.svg)](https://github.com/noah-hein/mazeGPT/commits/master)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/noah-hein/mazeGPT)](https://GitHub.com/noah-hein/mazeGPT/pull/)

#### Disclaimers
Does some maze generation and stuff. Working on this because I'm bored.
All this thing knows is mazes. By no means am I a master of machine learning. 
Hugging face and OpenAI are the ones to thank. They are doing the heaving lifting here.

Lots of inspiration from [NanoGPT](https://github.com/karpathy/nanoGPT) 
and [Andrej Apathy's](https://github.com/karpathy) 
great [video](https://www.youtube.com/watch?v=kCc8FmEb1nY).

|                                    |                                |
|:----------------------------------:|:------------------------------:|
| ![Inception](/media/inception.jpg) | ![Shining](/media/shining.jpg) |
|                                    |                                |

# 🔍 Table of Contents
* 🌅 [Introduction](#introduction)
* ⏩ [Getting Started](#-getting-started)
* 🎓 [Authors](#authors)
* 📗 [Overview](docs/OVERVIEW.md#-overview)
  * 🌌 [Why](docs/OVERVIEW.md#-why)
  * 📐 [Representation](docs/OVERVIEW.md#-representing-a-maze)
  * 📤 [Tokenizer](docs/OVERVIEW.md#-tokenizer)
* 🏢 [Structure](docs/STRUCTURE.md#-structure)

## 🌅 Introduction
["Attention Is All You Need"](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) 
was a ground break paper in the world of machine learning in 2017.
The idea of a transformer has dramatically helped reduced the train time while improving the consistency
of attention across long periods of recurrent generation. The company [OpenAI](https://openai.com/) has two models ChatGPT and DALL·E both implementing transformers to achieve 
incredible results. 

The objective of this research project was to implement a transformer model for generating mazes. 
While there are numerous existing maze algorithms that perform well, they tend to produce recurrent patterns despite 
being seeded randomly. The goal is to achieve mazes that are more random and chaotic in nature and mimic human behavior.

For more detailed information visit the [Overview](docs/OVERVIEW.md)

## ⏩ Getting Started
### Virtual Environment (Optional)
I recommend creating a virtual python environment to contain the dependencies for the project. Although this is not
required, it will make bootstrapping the project far easier.

Run the following to create an environment named venv
```bash
python -m venv venv       # Creates virtual env
.\venv\Scripts\activate   # Activate venv
```

### GPU Support (Optional)
This project primary uses Huggingface Transformers to handle configure a lot of the model / training logic.
Under the hood it is using PyTorch (this could be changed for Tensorflow). Torch must be installed to run the trainer.

By default, the CPU variant of PyTorch will be installed when bootstrapping. If you have a GPU I highly recommend
using it. Transformer models are not fast with a CPU.

To install the GPU variant vist [PyTorch Getting Started](https://pytorch.org/get-started/locally/)

```bash
# Example command for PyTorch CUDA 11.8 Windows
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Installing Dependencies
To install all the required dependencies run the following.
```bash
pip install .
```

### Config
Each script within the project requires the MazeAIConfig. This class can be extended and modified to suite your needs.
Visit the config module to see the default example.

### CLI
The main scrips are prepare.py, train.py, and sample.py. Each file can be run individually to preform there intended
operation, but pay attention to the configuration passed in.

Below is an example of the bottom of train.py
```python
from src.train import MazeAITrainer
from src.config.default import MazeAIConfig

# Pay attention to this!
if __name__ == '__main__':
    """Allows you to run the train script without the CLI"""
    MazeAITrainer(MazeAIConfig())
```
As you can see the main constructor for each script requries the MazeAIConfig.

For each project script has been wrapped in a very simplistic [CLI](/mazegpt.py).

## 🎓 Authors
- Noah Hein ([@noah-hein](https://github.com/noah-hein))





