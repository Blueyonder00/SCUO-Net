import scipy.io as sio
import numpy as np
import torch
import cv2

data_names = {"orl": "ORL_32x32",
              "coil20": "COIL20",
              "coil100": "COIL100",
              "yaleb": "YaleBCrop025"}


def get_data(data_name, device):


    # 读取数据
    if data_name == 'yaleb':
        data = sio.loadmat('./datasets/YaleBCrop025.mat')
        img = data['Y']
        I = []
        Label = []
        for i in range(img.shape[2]):
            for j in range(img.shape[1]):
                temp = np.reshape(img[:, j, i], [42, 48])
                temp = cv2.resize(temp, (32,32))
                Label.append(i)
                I.append(temp)
        I = np.array(I)
        labels = np.array(Label[:])
        features = I.reshape(-1, 1, 32,32)
        # features = np.expand_dims(features[:], 3)
    else:
        filename = "./datasets/" + data_names[data_name] + ".mat"
        data = sio.loadmat(filename)
        features, labels = data['fea'].reshape((-1, 1, 32, 32))[:2880,:], data['gnd'][:2880]

    # 数据简单处理
    features = torch.from_numpy(features).float().to(device)
    labels = np.squeeze(labels - np.min(labels))

    return features, labels

if __name__ == "__main__":
    for name in data_names.keys():
        if name in ["nuswide", "caltech20"]:
            continue

        x, y = get_data(name, 'cpu')
        print(name)
        print(x.shape)
        print(len(np.unique(y)))
