import scipy.io as sio
import numpy as np
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
import pandas as  pd
import  csv
from sklearn.cluster import KMeans
import numpy as np
pd.options.display.max_rows = 100000
pd.options.display.max_columns = 20
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
import pandas as pd
from sklearn import svm, preprocessing
# import tushare as ts
from sklearn.svm import SVR
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
data_names = {"orl": "ORL_32x32",
              "coil20": "COIL20",
              "coil100": "COIL100"}
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

def get_data(data_name, device):
    filename = "./datasets/" + data_names[data_name] + ".mat"

    # 读取数据
    data = sio.loadmat(filename)
    features, labels = data['fea'], data['gnd']

    # 数据简单处理
    data=features
    X_data1 = data[:,:]#聚类数据，可以选择行列，并转置

    Sta=StandardScaler()###加标准化，不想要标准化可以注释掉这三句
    Sta.fit(X=X_data1)####标准化用均值跟方差算的，对有个别极值的点不敏感
    X_data1=Sta.transform(X_data1)

    # MM=MinMaxScaler()#######最大最小归一化，现在是屏蔽的，想用这个可以取消屏蔽，这个和标准化只能选一个用
    # MM.fit(X=X_data1)#####最大最小归一化是用最大值和最小值算的，对极值比较敏感，这里可能不适用
    # X_data1=MM.transform(X_data1)
    # #
    pca = PCA(n_components=2)#选择降维后的维度数量
    pca = pca.fit(X_data1)#降维
    X_draw = pca.transform(X_data1)#执行降维
    # X_dr=X_data1
    kmeans = KMeans(n_clusters=20,init='k-means++', n_init=10,  max_iter=300, tol=0.0001,
           verbose=0,  random_state=None,  copy_x=True,   algorithm='auto'
           )#使用Kmeans聚类，聚2类，初始化方式k-means++，随机初始质心10，最大迭代次数300，相关的扰动在两次迭代的聚类中心，
    kmeans.fit(X_draw)#拟合数据
    y_kmeans = kmeans.predict(X_draw)#预测数据


    fig = plt.figure()#设置画布
    ax = fig.add_subplot(111)#设置画布属性

    xs =X_draw[:,0]#依次赋值x，y，z三个点坐标
    ys = X_draw[:,1]
    ax.scatter(xs, ys, c=y_kmeans, marker='o')#颜色用的预测的标签值

    output = kmeans.cluster_centers_#获得中心点坐标数组
    #output = MS.cluster_centers_#获得中心点坐标数组

    # print(output)
    xx =output[:,0]#依次复制x，y，z三个中心点坐标
    yx = output[:,1]
    ax.scatter(xx, yx,  c='black', marker='x',s=50,alpha=1)#画中心点
    ax.set_xlabel('xs')#设置坐标轴名字
    ax.set_ylabel('ys')



    return features, labels,y_kmeans


if __name__ == "__main__":
    for name in data_names.keys():
        if name in ["nuswide", "caltech20"]:
            continue

        x, y_true,y_pred = get_data(name, 'cpu')
        accuracy = acc(y_true,y_pred)
        plt.title(label="%s"%name)
        plt.show()  # 展示图像
        print(name,accuracy)
        print(x.shape)
        print(len(np.unique(y_true)))
