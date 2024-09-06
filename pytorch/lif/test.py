import numpy as np
import torch
from torch import nn
import lif


# class FastSigmoid(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input_, slope=25):
#         ctx.save_for_backward(input_)
#         ctx.slope = slope
#         out = (input_ > 0).float()
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         (input_,) = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad = grad_input / (ctx.slope * torch.abs(input_) + 1.0) ** 2
#         return grad, None


# def fast_sigmoid(slope=25):
#     slope = slope

#     def inner(x):
#         return FastSigmoid.apply(x, slope)

#     return inner


# class LIFSpike(nn.Module):
#     def __init__(self, thresh=0.5, tau=0.75, gamma=1.0, dspike=True, soft_reset=True):
#         """
#         Implementing the LIF neurons.
#         @param thresh: firing threshold;
#         @param tau: membrane potential decay factor;
#         @param gamma: hyper-parameter for controlling the sharpness in surrogate gradient;
#         @param dspike: whether using rectangular gradient of dspike gradient;
#         @param soft_reset: whether using soft-reset or hard-reset.
#         """
#         super(LIFSpike, self).__init__()
#         self.thresh = thresh
#         self.beta = tau
#         self.gamma = gamma
#         self.soft_reset = soft_reset
#         self._act = fast_sigmoid(slope=25)

#     def forward(self, x):
#         T = x.shape[2]
#         spike_out = [None] * T
#         mem = 0
#         for t in range(T):
#             mem = mem * self.beta + x[:, :, t]
#             spike = self._act(mem - self.thresh)
#             if self.soft_reset:
#                 mem -= spike * self.thresh
#             else:
#                 mem = (1 - spike) * mem
#             spike_out[t] = spike
#         return torch.stack(spike_out, dim=2)



class Lif(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = lif.lif_fw(input, thresh, beta)
        ctx.save_for_backward(input)
        return output

class Lif_ee(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  input, thresh, beta,prev, es_thresh):
        output, es = lif.lif_es_fw(input, prev, thresh, beta, es_thresh)
        ctx.save_for_backward(input)
        return output, es

def LIFSpike(thresh, beta):
    thresh = thresh
    beta = beta

    def inner(x):
        return Lif.apply(x)

    return inner

def LIFSpike_ee(thresh, beta):
    thresh = thresh
    beta = beta

    def inner(x, prev, es_thresh):
        return Lif_ee.apply(x, thresh, beta, prev, es_thresh)

    return inner

import time
if __name__ == '__main__':
    input = []
    for i in range(100):
        input.append(torch.randn(1, 9, 128))
    prev = torch.ones(128,dtype=torch.int32)

    
    thresh = 0.5
    beta = 0.75
    lifes = LIFSpike_ee(thresh, beta)

    t0 = time.time()
    net = LIFSpike(thresh, beta)
    for x in input:
        out = net(x)
    
    t1 = time.time()
    for x in input:
        out_lif, es = lifes(x, prev, 10000)
    t2 = time.time()
    
    print(torch.allclose(out_lif, out))
    print(es)
    tlast1 = round((t1 - t0) * 1e5, 5)
    tlast2 = round((t2 - t1) * 1e5, 5)
    
    print(f'P:{tlast1}, C:{tlast2}')