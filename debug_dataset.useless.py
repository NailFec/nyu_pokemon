import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# 确保使用修正后的 dataset.py 和 train.py 中的 transform 定义
from dataset import PokemonDataset
from utils import load_label_map
from train import train_transform  # 导入你在train.py中定义的transform

# --- 设置 ---
DATA_PATH = "pokemon/pokemon_train/train_data.npy"
LABEL_PATH = "pokemon/pokemon_train/train_labels.npy"
LABEL_MAP_PATH = "pokemon/types2label.txt"
BATCH_SIZE = 8


def imshow(inp, title=None):
    """用于显示图像的辅助函数"""
    inp = inp.numpy().transpose((1, 2, 0))
    # 逆归一化，以便我们能看清原始图像
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontsize=8, wrap=True)
    plt.pause(0.001)


def main():
    # 加载数据集和标签映射
    dataset = PokemonDataset(DATA_PATH, LABEL_PATH, transform=train_transform)
    _, label_to_type = load_label_map(LABEL_MAP_PATH)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 获取一个批次的数据
    inputs, classes = next(iter(loader))

    # 创建图像网格
    out = torchvision.utils.make_grid(inputs)

    # 获取标签字符串
    class_titles = [label_to_type[c.item()] for c in classes]

    # 显示图像
    plt.figure(figsize=(12, 6))
    imshow(out, title=', '.join(class_titles))
    plt.show()


if __name__ == "__main__":
    main()