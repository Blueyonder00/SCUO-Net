import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSamePad(nn.Module):
    """
    Implement Tensorflow's 'SAME' padding mode in Conv2d.
    When an odd number, say `m`, of pixels are need to pad, Tensorflow will pad one more column at right or one more
    row at bottom. But Pytorch will pad `m+1` pixels, i.e., Pytorch always pads in both sides.
    So we can pad the tensor in the way of Tensorflow before call the Conv2d module.
    """

    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)


class ConvTranspose2dSamePad(nn.Module):
    """
    This module implements the "SAME" padding mode for ConvTranspose2d as in Tensorflow. A tensor with width w_in,
    feed it to ConvTranspose2d(ci, co, kernel, stride), the width of output tensor T_nopad: w_nopad = (w_in - 1) *
    stride + kernel If we use padding, i.e., ConvTranspose2d(ci, co, kernel, stride, padding, output_padding),
    the width of T_pad: w_pad = (w_in - 1) * stride + kernel - (2*padding - output_padding) = w_nopad - (2*padding -
    output_padding) Yes, in ConvTranspose2d, more padding, the resulting tensor is smaller, i.e., the padding is
    actually deleting row/col. If `pad`=(2*padding - output_padding) is odd, Pytorch deletes more columns in the
    left, i.e., the first ceil(pad/2) and last `pad - ceil(pad/2)` columns of T_nopad are deleted to get T_pad. In
    contrast, Tensorflow deletes more columns in the right, i.e., the first floor(pad/2) and last `pad - floor(
    pad/2)` columns are deleted. For the height, Pytorch deletes more rows at top, while Tensorflow at bottom. In
    practice, we usually want `w_pad = w_in * stride`, i.e., the "SAME" padding mode in Tensorflow, so the number of
    columns to delete: pad = 2*padding - output_padding = kernel - stride We can solve the above equation and get:
    padding = ceil((kernel - stride)/2), and output_padding = 2*padding - (kernel - stride) which is either 1 or 0.
    But to get the same result with Tensorflow, we should delete values by ourselves instead of using padding and
    output_padding in ConvTranspose2d. To get there, we check the following conditions: If pad = kernel - stride is
    even, we can directly set padding=pad/2 and output_padding=0 in ConvTranspose2d. If pad = kernel - stride is odd,
    we can use ConvTranspose2d to get T_nopad, and then delete `pad` rows/columns by ourselves; or we can use
    ConvTranspose2d to delete `pad - 1` by setting `padding=(pad - 1) / 2` and `ouput_padding=0` and then delete the
    last row/column of the resulting tensor by ourselves. Here we implement the former case. This module should be
    called after the ConvTranspose2d module with shared kernel_size and stride values. And this module can only
    output a tensor with shape `stride * size_input`. A more flexible module can be found in `yaleb.py` which can
    output arbitrary size as specified.
    """

    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
        """
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)



        self.encoder = nn.Sequential()
        for i in range(1, len(channels)):
            #  Each layer will divide the size of feature map by 2
            self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module('conv%d' % i,
                                    nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=(2, 2)))
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        self.encoderO = nn.Sequential()
        for i in range(len(channels) - 1):
            self.encoderO.add_module('deconv%d' % (i + 1),
                                   nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i],
                                                      stride=(2, 2)))
            self.encoderO.add_module('padd%d' % i, ConvTranspose2dSamePad(kernels[i], 2))
            self.encoderO.add_module('relud%d' % i, nn.ReLU(True))
        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module('deconv%d' % (i + 1), nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=(2, 2)))
            self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(kernels[i], 2))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        # h2 = self.encoderO(x)
        # h = h1
        y = self.decoder(h)
        return y


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y
# sparse loss
# def sparse_loss(autoencoder,x):
#     model_list = list(autoencoder.ae.children())
#     values = x
#     loss = 0
#     for i in range(len(model_list[0])//3):
#         pad_layer = model_list[0][3*i]
#         conv_layer = model_list[0][3*i+1]
#         relu = model_list[0][3*i+2]
#         values = relu(conv_layer(pad_layer(values)))
#         # L1
#         # loss += torch.mean(torch.abs(values))
#         # L2
#         loss += torch.mean(torch.pow(values,2))
#     for i in range(len(model_list[1])//3):
#         conv_layer = model_list[1][3*i]
#         pad_layer = model_list[1][3*i+1]
#         relu = model_list[1][3*i+2]
#         values = relu(pad_layer(conv_layer(values)))
#         # loss += torch.mean(torch.abs(values))
#         loss += torch.mean(torch.pow(values, 2))
#     return loss


class DSCNet(nn.Module):
    def __init__(self, channels, kernels, num_sample):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.self_expression = SelfExpression(self.n)
        # self.self_expression2 = SelfExpression(self.n)
        # encoder layer1
        self.pad1 = Conv2dSamePad(kernels[0], 2)
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=kernels[0], stride=(2, 2))
        self.relu1 = nn.ReLU(True)
        # # encoder layer2
        # self.pad2 = Conv2dSamePad(kernels[1], 2)
        # self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=kernels[1], stride=(2, 2))
        # self.relu2 = nn.ReLU(True)
        # # encoder layer3
        # self.pad3 = Conv2dSamePad(kernels[2], 2)
        # self.conv3 = nn.Conv2d(channels[2], channels[3], kernel_size=kernels[2], stride=(2, 2))
        # self.relu3 = nn.ReLU(True)

        # overcomplete encoder layer1
        self.conv1_O = nn.ConvTranspose2d(channels[0], channels[1], kernel_size=kernels[0], stride=(2, 2))
        self.pad1_O = ConvTranspose2dSamePad(kernels[0], 2)
        self.relu1_O = nn.ReLU(True)
        # # overcomplete encoder layer2
        # self.conv2_O = nn.ConvTranspose2d(channels[1], channels[2], kernel_size=kernels[1], stride=(2, 2))
        # self.pad2_O = ConvTranspose2dSamePad(kernels[1], 2)
        # self.relu2_O = nn.ReLU(True)
        # # overcomplete encoder layer3
        # self.conv3_O = nn.ConvTranspose2d(channels[2], channels[3], kernel_size=kernels[2], stride=(2, 2))
        # self.pad3_O = ConvTranspose2dSamePad(kernels[2], 2)
        # self.relu3_O = nn.ReLU(True)

        self.intere1_1 = nn.Conv2d(channels[1], channels[1], kernels[0], stride=1, padding=1)
        self.inte1_1bn = nn.BatchNorm2d(channels[1])
        # self.intere2_1 = nn.Conv2d(channels[2], channels[2], kernels[1], stride=1, padding=1)
        # self.inte2_1bn = nn.BatchNorm2d(channels[2])
        # self.intere3_1 = nn.Conv2d(channels[3], channels[3], kernels[2], stride=1, padding=1)
        # self.inte3_1bn = nn.BatchNorm2d(channels[3])

        self.intere1_2 = nn.Conv2d(channels[1], channels[1], kernels[0], stride=1, padding=1)
        self.inte1_2bn = nn.BatchNorm2d(channels[1])
        # self.intere2_2 = nn.Conv2d(channels[2], channels[2], kernels[1], stride=1, padding=1)
        # self.inte2_2bn = nn.BatchNorm2d(channels[2])
        # self.intere3_2 = nn.Conv2d(channels[3], channels[3], kernels[2], stride=1, padding=1)
        # self.inte3_2bn = nn.BatchNorm2d(channels[3])

        # decoder
        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module('deconv%d' % (i + 1),
                                    nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i],
                                                       stride=(2, 2)))
            self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(kernels[i], 2))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))
    def forward(self, x):  # shape=[n, c, w, h]
        # layer1 out
        out = self.relu1(self.conv1(self.pad1(x)))
        out1 = self.relu1_O(self.pad1_O(self.conv1_O(x)))
        temp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte1_1bn(self.intere1_1(out1))),scale_factor=(0.25,0.25),
                        mode='bilinear'))
        out1 = torch.add(out1, F.interpolate(F.relu(self.inte1_2bn(self.intere1_2(temp))), scale_factor=(4, 4),
                                           mode='bilinear'))
        # # layer2 out
        # out = self.relu2(self.conv2(self.pad2(out)))
        # out1 = self.relu2_O(self.pad2_O(self.conv2_O(out1)))
        # temp = out
        # out = torch.add(out, F.interpolate(F.relu(self.inte2_1bn(self.intere2_1(out1))), scale_factor=(0.0625, 0.0625),
        #                                    mode='bilinear'))
        # out1 = torch.add(out1, F.interpolate(F.relu(self.inte2_2bn(self.intere2_2(temp))), scale_factor=(16, 16),
        #                                     mode='bilinear'))
        # # layer3 out
        # out = self.relu3(self.conv3(self.pad3(out)))
        # out1 = self.relu3_O(self.pad3_O(self.conv3_O(out1)))
        # # out = torch.add(out, F.interpolate(F.relu(self.inte3_1bn(self.intere3_1(out1))), scale_factor=(0.0625, 0.0625),
        # #                                    mode='bilinear'))
        # # out1 = torch.add(out1, F.interpolate(F.relu(self.inte3_2bn(self.intere3_2(temp))), scale_factor=(16, 16),
        # #                                     mode='bilinear'))

        z = F.max_pool2d(out1, kernel_size=4, stride=4) + out

        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        z = z.view(self.n, -1)  # shape=[n, d]
        z_recon = self.self_expression(z)  # shape=[n, d]
        z_recon_reshape = z_recon.view(shape)

        x_recon = self.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return x_recon, z, z_recon

    def loss_fn(self, x, x_recon, z, z_recon, weight_coe, weight_self_exp, weight_sim):
        m = 0.325
        r = 0.25
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coe = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        # loss_sparse = sparse_loss(self,x)
        # loss_coe = torch.sum(torch.abs(self.self_expression.Coefficient))
        loss_self_exp = F.mse_loss(z_recon, z, reduction='sum')
        contrastive_loss = 0
        for i in range(self.n):
            point = z[i].reshape(1, -1)
            similarity = F.cosine_similarity(point, z_recon, dim=1)
            pos_similarity = similarity[i]
            neg_similarity = torch.cat((similarity[:i], similarity[i + 1:]))
            contrastive_loss += (
                torch.log(1 + (torch.exp(r * (neg_similarity + m))).sum() * torch.exp(-r * pos_similarity))).item()
        loss = loss_ae + weight_coe * loss_coe + weight_self_exp * loss_self_exp + weight_sim * contrastive_loss

        return loss


class UONet(nn.Module):
    def __init__(self,num_sample,kernels,channels):
        super(UONet, self).__init__()
        self.n = num_sample
        self.self_expression = SelfExpression(self.n)

        self.encoder1 = nn.Conv2d(channels[0], channels[1], kernels[0], stride=1,
                                  padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB
        self.en1_bn = nn.BatchNorm2d(channels[1])
        self.encoder2 = nn.Conv2d(channels[1], channels[2], kernels[1], stride=1, padding=1)
        self.en2_bn = nn.BatchNorm2d(channels[2])
        self.encoder3 = nn.Conv2d(channels[2], channels[3], kernels[2], stride=1, padding=1)
        self.en3_bn = nn.BatchNorm2d(channels[3])

        self.decoder1 = nn.Conv2d(channels[3], channels[2], kernels[0], stride=1, padding=1)
        self.de1_bn = nn.BatchNorm2d(channels[2])
        self.decoder2 = nn.Conv2d(channels[2], channels[1], kernels[1], stride=1, padding=1)
        self.de2_bn = nn.BatchNorm2d(channels[1])
        self.decoder3 = nn.Conv2d(channels[1], channels[0], kernels[2], stride=1, padding=1)
        self.de3_bn = nn.BatchNorm2d(channels[0])


        self.encoderf1 = nn.Conv2d(channels[0], channels[1], kernels[0], stride=1,
                                   padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB
        self.enf1_bn = nn.BatchNorm2d(channels[1])
        self.encoderf2 = nn.Conv2d(channels[1], channels[2], kernels[1], stride=1, padding=1)
        self.enf2_bn = nn.BatchNorm2d(channels[2])
        self.encoderf3 = nn.Conv2d(channels[2], channels[3], kernels[2], stride=1, padding=1)
        self.enf3_bn = nn.BatchNorm2d(channels[3])

        self.intere1_1 = nn.Conv2d(channels[1], channels[1], kernels[0], stride=1, padding=1)
        self.inte1_1bn = nn.BatchNorm2d(channels[1])
        self.intere2_1 = nn.Conv2d(channels[2], channels[2], kernels[1], stride=1, padding=1)
        self.inte2_1bn = nn.BatchNorm2d(channels[2])
        self.intere3_1 = nn.Conv2d(channels[3], channels[3], kernels[2], stride=1, padding=1)
        self.inte3_1bn = nn.BatchNorm2d(channels[3])

        self.intere1_2 = nn.Conv2d(channels[1], channels[1], kernels[0], stride=1, padding=1)
        self.inte1_2bn = nn.BatchNorm2d(channels[1])
        self.intere2_2 = nn.Conv2d(channels[2], channels[2], kernels[1], stride=1, padding=1)
        self.inte2_2bn = nn.BatchNorm2d(channels[2])
        self.intere3_2 = nn.Conv2d(channels[3], channels[2], kernels[2], stride=1, padding=1)
        self.inte3_2bn = nn.BatchNorm2d(channels[3])

        # self.interd1_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # self.intd1_1bn = nn.BatchNorm2d(32)
        # self.interd2_1 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        # self.intd2_1bn = nn.BatchNorm2d(16)
        # self.interd3_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.intd3_1bn = nn.BatchNorm2d(64)
        #
        # self.interd1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # self.intd1_2bn = nn.BatchNorm2d(32)
        # self.interd2_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        # self.intd2_2bn = nn.BatchNorm2d(16)
        # self.interd3_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.intd3_2bn = nn.BatchNorm2d(64)

        # self.final = nn.Conv2d(8, 2, 1, stride=1, padding=0)

        # self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder 1
        out = F.relu(self.en1_bn(F.max_pool2d(self.encoder1(x), 2, 2)))  # U-Net branch
        out1 = F.relu(
            self.enf1_bn(F.interpolate(self.encoderf1(x), scale_factor=(2, 2), mode='bilinear')))  # Ki-Net branch
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.inte1_1bn(self.intere1_1(out1))), scale_factor=(0.25, 0.25),
                                           mode='bilinear'))  # CRFB-ki-net-branch
        out1 = torch.add(out1, F.interpolate(F.relu(self.inte1_2bn(self.intere1_2(tmp))), scale_factor=(4, 4),
                                             mode='bilinear'))  # CRFB-U-net-branch
        # skip connection
        # u1 = out  # skip conn
        # o1 = out1  # skip conn


        # encoder 2
        out = F.relu(self.en2_bn(F.max_pool2d(self.encoder2(out), 2, 2)))# U-Net branch
        out1 = F.relu(self.enf2_bn(F.interpolate(self.encoderf2(out1), scale_factor=(2, 2), mode='bilinear')))# Ki-Net branch
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.inte2_1bn(self.intere2_1(out1))), scale_factor=(0.0625, 0.0625),
                                           mode='bilinear')) # CRFB-ki-net-branch
        out1 = torch.add(out1, F.interpolate(F.relu(self.inte2_2bn(self.intere2_2(tmp))), scale_factor=(16, 16),
                                             mode='bilinear')) # CRFB-U-net-branch

        # u2 = out
        # o2 = out1

        # encoder 3
        out = F.relu(self.en3_bn(F.max_pool2d(self.encoder3(out), 2, 2)))
        out1 = F.relu(self.enf3_bn(F.interpolate(self.encoderf3(out1), scale_factor=(2, 2), mode='bilinear')))
        shape_out = out.shape
        # z1 = out.view(self.n, -1)  # shape=[n, d]
        shape_out1 = out1.shape
        # z2 = out1.view(self.n, -1)  # shape=[n, d]
        z = F.max_pool2d(out1, kernel_size=64, stride=64) + out
        z = z.view(self.n,-1)
        z_recon = self.self_expression(z)
        z_recon_reshape = z_recon.view(self.n, out.shape[1], out.shape[2], out.shape[3])

        ### End of encoder block

        ### Start Decoder

        # decoder1
        out = F.relu(self.de1_bn(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear')))  # U-NET
        # out1 = F.relu(self.def1_bn(F.max_pool2d(self.decoderf1(out1), 2, 2)))  # Ki-NET
        # tmp = out
        # out = torch.add(out, F.interpolate(F.relu(self.intd1_1bn(self.interd1_1(out1))), scale_factor=(0.0625, 0.0625),
        #                                    mode='bilinear'))
        # out1 = torch.add(out1, F.interpolate(F.relu(self.intd1_2bn(self.interd1_2(tmp))), scale_factor=(16, 16),
        #                                      mode='bilinear'))

        # decoder2
        out = F.relu(self.de2_bn(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear')))
        # out1 = F.relu(self.def2_bn(F.max_pool2d(self.decoderf2(out1), 2, 2)))
        # tmp = out
        # out = torch.add(out, F.interpolate(F.relu(self.intd2_1bn(self.interd2_1(out1))), scale_factor=(0.25, 0.25),
        #                                    mode='bilinear'))
        # out1 = torch.add(out1, F.interpolate(F.relu(self.intd2_2bn(self.interd2_2(tmp))), scale_factor=(4, 4),
        #                                      mode='bilinear'))

        # decoder3
        out = F.relu(self.de3_bn(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear')))
        # out1 = F.relu(self.def3_bn(F.max_pool2d(self.decoderf3(out1), 2, 2)))

        # out = torch.add(out, out1)  # fusion of both branches

        # out = F.relu(self.final(out))  # 1*1 conv

        # out = self.soft(out)
        # return out
        return out,z,z_recon
    def loss_pre_fn(self,x,x_recon):
        loss_ae = F.mse_loss(x_recon,x,reduction='sum')
        return loss_ae
    def loss_fn(self, x, x_recon, z, z_recon, weight_coe, weight_self_exp, weight_sim):
        t = 20
        m = 0.325
        r = 0.25
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coe = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        # loss_sparse = sparse_loss(self,x)
        # loss_coe = torch.sum(torch.abs(self.self_expression.Coefficient))
        loss_self_exp = F.mse_loss(z_recon, z, reduction='sum')
        contrastive_loss = 0
        for i in range(self.n):
            point = z[i].reshape(1, -1)
            similarity = F.cosine_similarity(point, z_recon, dim=1)
            pos_similarity = similarity[i]
            neg_similarity = torch.cat((similarity[:i], similarity[i + 1:]))
            contrastive_loss += (
                torch.log(1 + (torch.exp(r * (neg_similarity + m))).sum() * torch.exp(-r * pos_similarity))).item()
        loss = loss_ae + weight_coe * loss_coe + weight_self_exp * loss_self_exp + weight_sim * contrastive_loss

        return loss