from pynvml import *
from torch import cuda


def determine_train_device():
    device = "cuda" if cuda.is_available() else "cpu"
    cuda.empty_cache()
    print("Using (" + device + ") to train")


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def bordered(text):
    lines = text.splitlines()
    width = max(len(s) for s in lines)
    res = ['┌' + '─' * width + '┐']
    for s in lines:
        res.append('│' + (s + ' ' * width)[:width] + '│')
    res.append('└' + '─' * width + '┘')
    return '\n'.join(res)


def rooted(path: str):
    root_path = os.path.dirname(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(root_path, path))
