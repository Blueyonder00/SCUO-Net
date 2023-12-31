class ConfigCoil100:
    # 数据集参数
    num_cluster = 40
    num_sample = 2880

    # 网络参数
    channels = [1,50]
    kernels = [5]

    # 训练参数
    epochs = 200
    lr = 0.00045
    weight_coe = 1.0
    weight_self_exp = 1
    weight_sparse = 0.5
    weight_sim = 0.01

    # post clustering parameters
    alpha = 0.03  # threshold of C
    dim_subspace = 13  # dimension of each subspace
    ro = 8  #

    # 其它参数
    comment64 = False
    show_freq = 1
