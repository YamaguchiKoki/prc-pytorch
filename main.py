import os
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import matplotlib.pyplot as plt

def make_filepath_list():
    """
    学習データ、検証データそれぞれのファイルへのパスを格納したリストを作成する

    Returns
    ----------
    train_file_list : list
        学習データのファイルへのパスを格納したリスト
    valid_file_list : list
        検証データのファイルへのパスを格納したリスト
    """
    train_file_list = []
    valid_file_list = []

    for top_dir in os.listdir("./Images/"):
        file_dir = os.path.join("./Images/", top_dir)
        file_list = os.listdir(file_dir)

        num_data = len(file_list)
        num_split = int(num_data * 0.8)

        # 8割学習データ 2割検証データ
        train_file_list += [os.path.join('./Images', top_dir, file).replace('\\', '/') for file in file_list[:num_split]]
        valid_file_list += [os.path.join('./Images', top_dir, file).replace('\\', '/') for file in file_list[:num_split]]

        return train_file_list, valid_file_list

class ImageTransform(object):
    """
    入力画像の前処理
    画像のサイズをリサイズする

    Attributes
    ----------
    resize : int
        リサイズ先の画像の大きさ
    mean : (R, G, B)
        各色チャネルの平均値
    std : (R, G, B)
        各色チャネルの標準偏差
    """
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                # データオーギュメンテーション
                transforms.RandomHorizontalFlip(),
                # 画像をresize x resizeの大きさに統一
                transforms.Resize((resize, resize)),
                # テンソルに変換
                transforms.ToTensor(),
                # 色情報の標準化
                transforms.Normalize(mean, std)
            ]),
            'valid': transforms.Compose([
                # 画像をresize x resizeの大きさに統一
                transforms.Resize((resize, resize)),
                # テンソルに変換
                transforms.ToTensor(),
                # 色情報の標準化
                transforms.Normalize(mean, std)
            ])
        }
    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'valid'
            前処理のモードを指定
        """
        return self.data_transform[phase](img)


if __name__ == "__main__":
    img = Image.open('./Images/n02085620-Chihuahua/n02085620_199.jpg')
    resize = 300

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    transform = ImageTransform(resize, mean, std)
    img_transformed = transform(img, 'train')

    plt.imshow(img_transformed.numpy().transpose((1, 2, 0)))
    plt.show()
