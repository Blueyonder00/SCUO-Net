import numpy as np
import torch
from torch import optim
import logging
import matplotlib.pyplot as plt
from config import get_config
from load_data import get_data
from network import DSCNet
from network import UONet
from util import load_model, save_mode
from post_clustering import spectral_clustering
from evaluate import get_score
from sklearn.preprocessing import normalize
import warnings

def shrink_matrix(matrix,block_size,shrink_factor):
    shrunk_matrix = np.copy(matrix)
    rows, cols = shrunk_matrix.shape

    for i in range(rows):
        for j in range(cols):
            if (i < block_size and j < block_size) or (i >= rows - block_size and j >= cols - block_size):

                shrunk_matrix[i,j] -= shrink_factor
    return shrunk_matrix
class RunModel:
    def __init__(self, name):
        # get configs
        self.cfg = get_config(name)
        # get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # get dataloader
        self.features, self.labels = get_data(name, self.device)
        # set name
        self.name = name

    def get_param(self):
        cfg = self.cfg
        return cfg.epochs, cfg.weight_coe, cfg.weight_self_exp, cfg.weight_sim, cfg.num_cluster, cfg.dim_subspace, cfg.alpha, cfg.ro, cfg.comment64

    def train_dsc(self):
        # 模型参数
        cfg = self.cfg
        model = DSCNet(num_sample=cfg.num_sample, channels=cfg.channels, kernels=cfg.kernels).to(self.device)
        # model = UONet(num_sample=cfg.num_sample,channels=cfg.channels, kernels=cfg.kernels).to(self.device)
        # try:
        #     load_model(model.ae, self.name, 'pretrained_weights_original')
        # except:
        #     warnings.warn('cannot load all parameters')
        try:
            model.load_state_dict(torch.load('/home/htz/szh-dsc/DscNet_reimplement/pretrain_models_szh/{}_pretrain_path.pth'.format(self.name)))
            print('load pretrain succeed')
        except:
            warnings.warn('cannot load all parameters')
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        epochs, weight_coe, weight_self_exp, weight_sim, num_cluster, dim_subspace, alpha, ro, comment64 = self.get_param()
        x = self.features
        y = self.labels
        max_acc = 0
        max_nmi = 0
        for epoch in range(epochs):
            # x_recon, z1_recon, z2_recon, z1, z2 = model(x)
            # pretrain
            # x_recon,z = model(x)
            # loss = model.loss_pre_fn(x_recon=x_recon,x=x)
            x_recon,z, z_recon = model(x)
            # loss = model.loss_fn(x, x_recon, z1_recon, z2_recon, z1,z2,weight_coe=weight_coe, weight_self_exp=weight_self_exp)
            loss = model.loss_fn(x, x_recon, z, z_recon,weight_coe=weight_coe, weight_self_exp=weight_self_exp,weight_sim=weight_sim)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if (epoch % 1 == 0 or epoch == epochs - 1) and epoch >= 0:
                # coe1 = model.self_expressionk.Coefficient.detach().to('cpu').numpy()
                # coe = model.self_expressionk.Coefficient.detach().to('cpu').numpy()
                # coe = 0.5 * coe1 + 0.5 * coe2
                coe = model.self_expression.Coefficient.detach().to('cpu').numpy()
                y_pred = spectral_clustering(coe, num_cluster, dim_subspace, alpha, ro, comment64)
                coe = normalize(coe).astype(np.float32)
                Aff = 0.5 * (np.abs(coe) + np.abs(coe).T)
                np.fill_diagonal(Aff,Aff.diagonal()-0.2)
                # 调整

                acc, nmi = get_score(y, y_pred)
                if acc > max_acc:
                    max_acc = acc
                    max_nmi = nmi
                if max_acc > 0.9:
                    torch.save(model.state_dict(),'./pretrain_models_szh/{}_pretrain_path.pth'.format(self.name))
                if max_acc > 0.97:
                   np.savetxt('y_pred_{}.csv'.format(self.name),y_pred)
                # 画affinity-matrix
            # if epoch % 10 == 0:
            #     plt.imshow(Aff, cmap='jet', vmax=0.4, vmin=0.05)
            #     plt.colorbar()
            #     plt.savefig('results/C_visual/COIL-20-{}.png'.format(epoch+100))
            #     plt.show()


                print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' % (epoch, loss.item() / y_pred.shape[0], acc, nmi))
        print('max_acc: %.4f, max_nmi: %.4f' % (max_acc,max_nmi))
            # print('Epoch %02d: loss=%.4f' % (epoch, loss.item()))
            # torch.save(model.state_dict(), './pretrain_models_szh/{}_pretrain_path.pth'.format(self.name))
        # save_mode(model, self.name)


if __name__ == "__main__":
    print("a")
    a = RunModel("orl")
    print(len(a.features))
    print(a.features[0].shape, a.features[1].shape, a.features[0].dtype)
