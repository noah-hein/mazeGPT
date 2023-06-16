import os
import numpy as np

DATA_DIR = "data"
TRAIN_FILENAME = "train.bin"
VALIDATION_FILENAME = "validation.bin"


def load_maze_data():
    data_dir = os.path.join(DATA_DIR)
    train_path = os.path.join(data_dir, TRAIN_FILENAME)
    validation_path = os.path.join(data_dir, VALIDATION_FILENAME)
    return (
        np.memmap(train_path, dtype=np.uint16, mode='r'),
        np.memmap(validation_path, dtype=np.uint16, mode='r')
    )


if __name__ == '__main__':
    (train_data, validation_data) = load_maze_data()
    print(validation_data)

