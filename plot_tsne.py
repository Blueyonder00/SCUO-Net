import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scipy.io as sio
from scipy.optimize import linear_sum_assignment as linear_assignment

data_names = {"orl": "ORL_32x32",
              "coil20": "COIL20",
              "coil100": "COIL100"}
data_name = 'coil20'
filename = "./datasets/" + data_names[data_name] + ".mat"

# 读取数据
data = sio.loadmat(filename)
features = data['fea'].reshape((-1, 32*32))
labels_TRUE = np.loadtxt('y_pred_{}'.format(data_name)+'.csv')
labels = np.loadtxt("/home/htz/szh-dsc/Self-Expressive-Network-szh/COIL-20_result/COIL-20_ypred_1440.csv")
# labels = data['gnd']-1
# tsne = TSNE(n_components=2,random_state=2).fit(features)
# x_tsne = tsne.fit_transform(features)
# x_min,x_max = x_tsne.min(0),x_tsne.max(0)
# x_norm = (x_tsne-x_min)/(x_max-x_min)
# plt.figure(figsize=(10,10))
# colors = plt.cm.rainbow(np.linspace(0,1,20))
# for i in range(len(x_norm)):
#     plt.text(x_norm[i,0],x_norm[i,1],'.',color=colors[int(labels[i,0])],fontdict={'weight':'bold','size': 40})
# # plt.scatter(data_tsne[:,0],data_tsne[:,1],c=labels)
# # plt.xticks([])
# # plt.yticks([])
# plt.title('{}'.format(data_name)+' clustering result')
# plt.show()

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind_row, ind_col = linear_assignment(np.max(w) - w)        # w.max()
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size
print(acc(labels_TRUE,labels))