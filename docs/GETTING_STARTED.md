## 🧠 Getting Started

### Installation
Below are the steps you should follow in order to set up the project environment.
If you are familiar with venv and PyTorch the installation section can be skipped.

#### Virtual Environment (Optional)
For this project, it's advisable to create a Python virtual environment to manage the project's dependencies. 
While this is not mandatory, it significantly simplifies the process of initializing the project.

To create a virtual environment named 'venv', execute the command shown below.
```bash
$ python -m venv venv       # Creates virtual env
$ .\venv\Scripts\activate   # Activate venv
```

#### GPU Support (Optional)
This project primarily utilizes Huggingface Transformers for the configuration of numerous model and training logic 
aspects. The underlying framework is PyTorch, although it could be substituted with Tensorflow if preferred. 
It's crucial to have PyTorch installed to operate the trainer.

Upon setup, the CPU variant of PyTorch is the default installation. However, if you have access to a GPU, its use is 
strongly recommended. Due to their intensive computational requirements, Transformer models perform slowly when executed 
on a CPU.

To install the GPU variant vist [PyTorch Getting Started](https://pytorch.org/get-started/locally/)

```bash
# Example command for PyTorch CUDA 11.8 Windows
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Installing Dependencies
To install all the required dependencies run the following.
```bash
$ pip install .
```

### Usage

#### Config
Every script in the project relies on the `MazeAIConfig` class. This class is customizable and can be extended according 
to your requirements. To view the default example, navigate to the config module.

When creating a new configuration, remember to add it to the AVAILABLE_CONFIGS dictionary in [mazegpt.py](/mazegpt.py). 
Doing so ensures that the configuration becomes accessible to the [CLI](#cli).

```python
AVAILABLE_CONFIGS: dict[str, Type[MazeAIConfig]] = {
    "default": MazeAIConfig,
    "foobar": FoobarConfig
}
```

#### Scripts
The main scrips are prepare.py, train.py, and sample.py. Each file can be run individually to preform there intended
operation, but pay attention to the configuration passed in.

Below is an example of the bottom of prepare.py
```python
if __name__ == '__main__':
    """Allows you to run the train script without the CLI"""
    MazeAIData(MazeAIConfig())
```
The main method allows you to run the script invidually, but requires manual modification of the config.
As stated above you can extend the configuration and provide your own.

To run the modules simply do
```bash
# Examples
$ python -m src.prepare
$ python -m src.train
$ python -m src.sample
```

#### CLI
For simplicity each project script has been wrapped in a simplistic [CLI](/mazegpt.py) from the 
[Click](https://click.palletsprojects.com/en/8.1.x/) library. The benefit from doing this is the ability to select
different configurations via the CLI per script.

To run the CLI use the following command
```bash
$ python mazegpt.py
```