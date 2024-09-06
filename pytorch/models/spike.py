
import torch
import lif
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.backbones import FCN, RESNET, DeepConvLSTM
from torch.onnx import symbolic_helper

class AvgMeter:

    def __init__(self):
        self.value = 0
        self.number = 0

    def add(self, v, n):
        self.value += v
        self.number += n

    def avg(self):
        if self.number == 0:
            self.number = 1
        return self.value / self.number

#*===================================================
#* Lif operators as torch.autograd.Function.
#*===================================================

class Lif(torch.autograd.Function):
    """ Lif operator.
    """
    @staticmethod
    def symbolic(g, input, thresh, beta):
        return g.op("custom::Lif", input, thresh_f=thresh, beta_f=beta) # '_f' is a symbolic suffix, the actual params are still 'thresh'/'beta'

    @staticmethod
    def forward(ctx, input, thresh, beta):
        output = lif.lif_fw(input, thresh, beta)
        ctx.save_for_backward(input)
        return output

class Lifee_Onnx(torch.autograd.Function):
    """ Lif with early exit for exporting to onnx model.
    !This Function contains an incomplete opertor for pyTorch inference.
    """
    @staticmethod
    def symbolic(g, input, thresh, beta, ee_thresh):
        return g.op("custom::Lifee", input, thresh_f=thresh, beta_f=beta, ee_thresh_f=ee_thresh)

    @staticmethod
    def forward(ctx, input, thresh, beta, ee_thresh):
        output = lif.lifee_onnx_fw(input, thresh, beta, ee_thresh)
        ctx.save_for_backward(input)
        return output
    
class Lifee(torch.autograd.Function):
    """ Lif with early exit for pyTorch inferencing.
    """
    @staticmethod
    def symbolic(g, input, thresh, beta, ee_thresh):
        return g.op("custom::Lifee", input, thresh_f=thresh, beta_f=beta, ee_thresh_f=ee_thresh)

    @staticmethod
    def forward(ctx, input, thresh, beta, prev, ee_thresh):
        output, spk_sum, ee = lif.lifee_fw(
            input, prev, thresh, beta, ee_thresh)
        ctx.save_for_backward(input)
        return output, spk_sum, ee

#*===================================================
#* Lif layers as nn.Module.
#*===================================================

class LifSpike(nn.Module):
    def __init__(self, thresh, beta):
        super(LifSpike, self).__init__()
        self.thresh = thresh
        self.beta = beta
    def forward(self, x):
        return Lif.apply(x, self.thresh, self.beta)

class LifeeSpike_Onnx(nn.Module):
    """ Lif with early exit for exporting to onnx model.
    !This Function contains an incomplete opertor for pyTorch inference.
    """
    def __init__(self, thresh, beta, ee_thresh):
        super(LifeeSpike_Onnx, self).__init__()
        self.thresh = thresh
        self.beta = beta
        self.ee_thresh = ee_thresh

    def forward(self, x):
        output = Lifee_Onnx.apply(x, self.thresh, self.beta, self.ee_thresh)
        return output
    
class LifeeSpike(nn.Module):
    def __init__(self, thresh, beta, ee_thresh):
        super(LifeeSpike, self).__init__()
        self.thresh = thresh
        self.beta = beta
        self.ee_thresh = ee_thresh
        self.prev_spk_sum = None

    def forward(self, x):
        ee = False
        spk_sum = None
        output = None
        if self.prev_spk_sum is None:
            self.prev_spk_sum = torch.zeros(x.shape, dtype=torch.int32)
            output, spk_sum, _ = Lifee.apply(
                x, self.thresh, self.beta, self.prev_spk_sum, self.ee_thresh)
        else:
            output, spk_sum, ee = Lifee.apply(
                x, self.thresh, self.beta, self.prev_spk_sum, self.ee_thresh)
        self.prev_spk_sum = spk_sum
        return output, bool(ee)


#*===================================================
#* Layers Fused with Conv and Lif.
#*===================================================
import math

def unsqueeze_all(t):
    # Helper function to ``unsqueeze`` all the dimensions that we reduce over
    return t[None, :, None, None]

class FusedConvBN2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, conv_weight, eps=1e-3):
        assert X.ndim == 4  # N, C, H, W
        # (1) Only need to save this single buffer for backward!
        ctx.save_for_backward(X, conv_weight)

        # (2) Exact same Conv2D forward from example above
        X = F.conv2d(X, conv_weight)
        # (3) Exact same BatchNorm2D forward from example above
        sum = X.sum(dim=(0, 2, 3))
        var = X.var(unbiased=True, dim=(0, 2, 3))
        N = X.numel() / X.size(1)
        sqrt_var = torch.sqrt(var)
        ctx.eps = eps
        ctx.sum = sum
        ctx.N = N
        ctx.sqrt_var = sqrt_var
        mean = sum / N
        denom = sqrt_var + eps
        # Try to do as many things in-place as possible
        # Instead of `out = (X - a) / b`, doing `out = X - a; out /= b`
        # avoids allocating one extra NCHW-sized buffer here
        out = X - unsqueeze_all(mean)
        out /= unsqueeze_all(denom)
        return out

class FusedConv1dBNLif(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, thresh, beta, exp_avg_factor=0.1,
                 eps=1e-3, device=None, dtype=None):
        super(FusedConv1dBNLif, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Conv parameters
        weight_shape = (out_channels, in_channels, kernel_size)
        self.conv_weight = nn.Parameter(torch.empty(*weight_shape, **factory_kwargs))
        # Batch norm parameters
        num_features = out_channels
        self.num_features = num_features
        self.eps = eps
        self.thresh = thresh
        self.beta = beta
        # Initialize
        self.reset_parameters()

    @staticmethod
    def symbolic(g, input, in_channels, out_channels, kernel_size, thresh, beta, exp_avg_factor,
                 eps):
        return g.op("custom::ConvLif", input, in_channels_i, out_channels_i, kernel_size_i, thresh_f, beta_f, exp_avg_factor_f,
                 eps_f) # '_f' is a symbolic suffix, the actual params are still 'thresh'/'beta'

    def forward(self, X):
        x = FusedConvBN2DFunction.apply(X, self.conv_weight, self.eps)
        x = Lif.apply(x, self.thresh, self.beta)
        return x

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))

#*===================================================
#* Models with Lif.
#*===================================================

class SFCN(FCN):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True, **kwargs):
        super(SFCN, self).__init__(n_channels,
                                   n_classes, out_channels, backbone)
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=8,
                      stride=1, padding=4, bias=False),
            nn.BatchNorm1d(32),
            LIFSpike(**kwargs)
        )
        self.conv_block2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(64),
            LIFSpike(**kwargs)
        )
        self.conv_block3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, out_channels, kernel_size=8,
                      stride=1, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            LIFSpike(**kwargs)
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x

class SFCN_eval(FCN):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True, **kwargs):
        super(SFCN_eval, self).__init__(n_channels,
                                   n_classes, out_channels, backbone)
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=8,
                      stride=1, padding=4, bias=False),
            nn.BatchNorm1d(32),
            LifSpike(thresh=kwargs['thresh'], beta=kwargs['tau'])
        )
        self.conv_block2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(64),
            LifSpike(thresh=kwargs['thresh'], beta=kwargs['tau'])
        )
        self.conv_block3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, out_channels, kernel_size=8,
                      stride=1, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            LifSpike(thresh=kwargs['thresh'], beta=kwargs['tau'])
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x
    
class SFCN_ONNX(FCN):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True, **kwargs):
        super(SFCN_ONNX, self).__init__(n_channels,
                                   n_classes, out_channels, backbone)
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=8,
                      stride=1, padding=4, bias=False),
            nn.BatchNorm1d(32),
            LifSpike(thresh=kwargs['thresh'], beta=kwargs['tau'])
        )
        self.conv_block2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(64),
            LifSpike(thresh=kwargs['thresh'], beta=kwargs['tau'])
        )
        self.conv_block3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, out_channels, kernel_size=8,
                      stride=1, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            LifSpike(thresh=kwargs['thresh'], beta=kwargs['tau'])
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits

import pickle
global_inferrence_count = 0
class SFCN_getspike(FCN):
    def __init__(self, n_channels, n_classes, save_path, out_channels=128, backbone=True, **kwargs):
        super(SFCN_getspike, self).__init__(n_channels,
                                   n_classes, out_channels, backbone)
        self.save_path = save_path
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=8,
                      stride=1, padding=4, bias=False),
            nn.BatchNorm1d(32),
            LifSpike(thresh=kwargs['thresh'], beta=kwargs['tau'])
        )
        self.conv_block2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(64),
            LifSpike(thresh=kwargs['thresh'], beta=kwargs['tau'])
        )
        self.conv_block3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, out_channels, kernel_size=8,
                      stride=1, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            LifSpike(thresh=kwargs['thresh'], beta=kwargs['tau'])
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        spike = []
        x = self.conv_block1(x)
        spike.append(np.array(x))
        x = self.conv_block2(x)
        spike.append(np.array(x))
        x = self.conv_block3(x)
        spike.append(np.array(x))
        global global_inferrence_count
        with open(f'{self.save_path}/{global_inferrence_count}.pkl', 'wb') as file:
            pickle.dump(spike, file)
        global_inferrence_count  += 1
        x = self.pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x

class SRESNET(RESNET):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True, **kwargs):
        super(RESNET, self).__init__(n_channels,
                                   n_classes, out_channels, backbone)
        
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=8,
                      stride=1, padding='same', bias=False),
            nn.BatchNorm1d(32),
            # LifSpike(thresh=kwargs['thresh'], beta=kwargs['tau'])
            LIFSpike(**kwargs)
        )

        # self.resblock = nn.Sequential(
        #     nn.Conv1d(n_channels, 32, kernel_size=8,
        #               stride=1, padding='same', bias=False),
        #     nn.BatchNorm1d(32),
        #     # LifSpike(thresh=kwargs['thresh'], beta=kwargs['tau'])
        #     LIFSpike(**kwargs)
        # )

        
        self.conv_block2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding='same', bias=False),
            nn.BatchNorm1d(64),
            # LifSpike(thresh=kwargs['thresh'], beta=kwargs['tau'])
            LIFSpike(**kwargs)
        )
        self.conv_block3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(64, 64, kernel_size=8, stride=1, padding='same', bias=False),
            nn.BatchNorm1d(64),
            # LifSpike(thresh=kwargs['thresh'], beta=kwargs['tau'])
            LIFSpike(**kwargs)
        )
        self.conv_block4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(64, 64, kernel_size=8, stride=1, padding='same', bias=False),
            nn.BatchNorm1d(64),
            # LifSpike(thresh=kwargs['thresh'], beta=kwargs['tau'])
            LIFSpike(**kwargs)
        )
        self.conv_block_l = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, out_channels, kernel_size=8,
                      stride=1, padding='same', bias=False),
            nn.BatchNorm1d(out_channels),
            # LifSpike(thresh=kwargs['thresh'], beta=kwargs['tau'])
            LIFSpike(**kwargs)
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

    def forward(self, x_in):
        x = x_in
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block_l(x)
        x = self.pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x
        # return logits

class CASNN(FCN):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True, ee_thresh=[0, 0, 0], **kwargs):
        super(CASNN, self).__init__(n_channels,
                                    n_classes, out_channels, backbone)
        self.thresh = kwargs['thresh']
        self.beta = kwargs['tau']
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=8,
                      stride=1, padding=4, bias=False),
            nn.BatchNorm1d(32)
        )
        self.conv_block2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(64)
        )
        self.conv_block3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, out_channels, kernel_size=8,
                      stride=1, padding=4, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.lifee1 = LifeeSpike(self.thresh, self.beta, ee_thresh[0])
        self.lifee2 = LifeeSpike(self.thresh, self.beta, ee_thresh[1])
        self.lifee3 = LifeeSpike(self.thresh, self.beta, ee_thresh[2])

    def forward(self, x_in):
        x = x_in
        x = self.conv_block1(x_in)
        x, ee = self.lifee1(x)
        if ee:
            return None, 0
        x = self.conv_block2(x)
        x, ee = self.lifee2(x)
        if ee:
            return None, 1
        x = self.conv_block3(x)
        x, ee = self.lifee3(x)
        if ee:
            return None, 2
        x = self.pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x
        # return logits


class CASNN_ONNX(FCN):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True, ee_thresh=[0, 0, 0], **kwargs):
        super(CASNN_ONNX, self).__init__(n_channels,
                                    n_classes, out_channels, backbone)
        self.thresh = kwargs['thresh']
        self.beta = kwargs['tau']
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=8,
                      stride=1, padding=4, bias=False),
            nn.BatchNorm1d(32)
        )
        self.conv_block2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(64)
        )
        self.conv_block3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, out_channels, kernel_size=8,
                      stride=1, padding=4, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.lifee1 = LifeeSpike_Onnx(self.thresh, self.beta, ee_thresh[0])
        self.lifee2 = LifeeSpike_Onnx(self.thresh, self.beta, ee_thresh[1])
        self.lifee3 = LifeeSpike_Onnx(self.thresh, self.beta, ee_thresh[2])

    def forward(self, x_in):
        x = x_in
        x = self.conv_block1(x_in)
        x = self.lifee1(x)
        x = self.conv_block2(x)
        x = self.lifee2(x)
        x = self.conv_block3(x)
        x = self.lifee3(x)
        x = self.pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits

class SDCL(DeepConvLSTM):

    def __init__(self, n_channels, n_classes, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True, **snn_p):
        super(SDCL, self).__init__(n_channels, n_classes,
                                   conv_kernels, kernel_size, LSTM_units, backbone)
        self.act1 = LIFSpike(**snn_p)
        self.act2 = LIFSpike(**snn_p)
        self.act3 = LIFSpike(**snn_p)
        self.act4 = LIFSpike(**snn_p)

        self.bn1 = nn.BatchNorm2d(conv_kernels)
        self.bn2 = nn.BatchNorm2d(conv_kernels)
        self.bn3 = nn.BatchNorm2d(conv_kernels)
        self.bn4 = nn.BatchNorm2d(conv_kernels)

        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.unsqueeze(1)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[-1, :, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        out = (input > 0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        # tmp = torch.ones_like(input)
        # tmp = torch.where(input.abs() < 0.5, 1., 0.)
        grad_input = grad_input * tmp
        return grad_input, None


class DSPIKE(nn.Module):
    def __init__(self, region=1.0):
        super(DSPIKE, self).__init__()
        self.region = region

    def forward(self, x, temp):
        out_bp = torch.clamp(x, -self.region, self.region)
        out_bp = (torch.tanh(temp * out_bp)) / \
                 (2 * np.tanh(self.region * temp)) + 0.5
        out_s = (x >= 0).float()
        return (out_s.float() - out_bp).detach() + out_bp


class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, slope=25):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.slope * torch.abs(input_) + 1.0) ** 2
        return grad, None


def fast_sigmoid(slope=25):
    slope = slope

    def inner(x):
        return FastSigmoid.apply(x, slope)
    return inner


class LIFSpike(nn.Module):
    def __init__(self, thresh=0.5, tau=0.75, gamma=1.0, dspike=True, soft_reset=True):
        """
        Implementing the LIF neurons.
        @param thresh: firing threshold;
        @param tau: membrane potential decay factor;
        @param gamma: hyper-parameter for controlling the sharpness in surrogate gradient;
        @param dspike: whether using rectangular gradient of dspike gradient;
        @param soft_reset: whether using soft-reset or hard-reset.
        """
        super(LIFSpike, self).__init__()
        dspike = True
        if not dspike:
            self.act = ZIF.apply
        else:
            # using the surrogate gradient function from Dspike:
            # https://proceedings.neurips.cc/paper/2021/file/c4ca4238a0b923820dcc509a6f75849b-Paper.pdf
            self.act = DSPIKE(region=1.0)

        self.thresh = thresh
        self.beta = tau
        self.gamma = gamma
        self.soft_reset = soft_reset
        self._act = fast_sigmoid(slope=25)

    def forward(self, x):
        mem = 0
        spike_out = []
        T = x.shape[2]
        for t in range(T):
            mem = mem * self.beta + x[:, :, t]
            spike = self._act(mem - self.thresh)
            mem = mem - spike * \
                self.thresh if self.soft_reset else (1 - spike) * mem
            spike_out.append(spike)
        return torch.stack(spike_out, dim=2)
