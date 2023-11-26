class ConfigOrl:
    # 数据集参数
    num_cluster = 40
    num_sample = 400

    # 网络参数
    channels = [1, 3, 5, 10]
    kernels = [3, 3, 3]

    # 训练参数
    epochs = 2000
    lr = 0.001
    weight_coe = 2.0
    weight_self_exp = 0.2
    weight_sim = 0.01
    # post clustering parameters
    alpha = 0.2  # threshold of C
    dim_subspace = 3  # dimension of each subspace
    ro = 1  #

    # 其它参数
    comment64 = True
    show_freq = 1
