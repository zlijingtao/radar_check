import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_centroid(input, grain_size, num_bits, M2D, half_lvls):
    if len(input.size()) == 2:
        original_size = input.size()
        reshaped_input = input.view(1, 1, original_size[0], original_size[1])
        pooling_result = F.avg_pool2d(reshaped_input, grain_size, grain_size)
        pooling_result = get_quantized(pooling_result, num_bits, M2D, half_lvls)
        pooling_result = pooling_result.view(pooling_result.size()[2:])
        pooling_result = pooling_result.unsqueeze(1).repeat(1,grain_size[0], 1).view(-1,pooling_result.size()[1]).transpose(0,1)
        output = pooling_result.repeat(1, grain_size[1]).view(-1,pooling_result.size()[1]).transpose(0,1)
    if len(input.size()) == 4:
        original_size = input.size()
        reshaped_input = input.permute(1, 2, 3, 0).view(1, 1, -1, original_size[0])
        pooling_result = F.avg_pool2d(reshaped_input, grain_size, grain_size)
        pooling_result = get_quantized(pooling_result, num_bits, M2D, half_lvls)
        pooling_result = pooling_result.view(pooling_result.size()[2:])
        pooling_result = pooling_result.unsqueeze(1).repeat(1,grain_size[0], 1).view(-1,pooling_result.size()[1]).transpose(0,1)
        pooling_result = pooling_result.repeat(1, grain_size[1]).view(-1,pooling_result.size()[1]).transpose(0,1)
        output = pooling_result.view(original_size[1], original_size[2], original_size[3], original_size[0]).permute(3, 0, 1, 2)
    return output
    
def get_quantized(input, num_bits, M2D, half_lvls):
    output = input.clone()
    if (M2D != 0.0) and (num_bits != 0):
        qmin = 0
        qmax = qmin + 2.**num_bits - 1.
        scale = 2 * half_lvls  * M2D / (qmax - qmin)
        output.div_(scale)
        output.add_((qmax - qmin)/2)
        output.clamp_(qmin, qmax).round_()
        output.add_(-(qmax - qmin)/2)
        output.mul_(scale).round_()
    else:
        output = input.clone().zero_()
    return output

class _quantize_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, step_size, half_lvls):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.step_size = step_size
        ctx.half_lvls = half_lvls
        output = F.hardtanh(input,
                            min_val=-ctx.half_lvls * ctx.step_size.item(),
                            max_val=ctx.half_lvls * ctx.step_size.item())

        output = torch.round(output / ctx.step_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.step_size

        return grad_input, None, None


quantize = _quantize_func.apply


class Unite(torch.autograd.Function):
    def __init__(self, grain_size, num_bits, M2D, step_size, half_lvls, save_path):
        super(Unite,self).__init__()
        self.grain_size = grain_size #grain size in tuple
        self.M2D = M2D
        self.num_bits = num_bits
        self.step_size = step_size
        self.half_lvls = half_lvls
        self.save_path = save_path
    def forward(self, input):
        self.save_for_backward(input)
        input = input/self.step_size
        self.centroid = get_centroid(input, self.grain_size, self.num_bits, self.M2D, self.half_lvls) 
        global ti
        global num_res
        ti += 1
        input_d = (input - self.centroid)
        output = input.clone().zero_()
        self.W = np.round((1-self.M2D) * self.half_lvls)
        output = input_d.clamp_(-self.W, self.W)
        if ti <=num_res:
            torch.save(self.centroid* self.step_size, self.save_path + '/saved_tensors/centroid{}.pt'.format(ti))
            torch.save(output* self.step_size, self.save_path + '/saved_tensors/deviation{}.pt'.format(ti))
        output = output + self.centroid

        return output

    def backward(self, grad_output):
        grad_input = grad_output.clone() / self.step_size
        input, = self.saved_tensors
        grad_input[input.ge(self.half_lvls * self.step_size.item())] = 0
        grad_input[input.le(-self.half_lvls * self.step_size.item())] = 0
        return grad_input

class quan_Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True, 
                 grain_size = (1,1), num_bits = 3, M2D = 0.0, save_path = './'):
        super(quan_Conv2d, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)
        self.N_bits = 8
        self.grain_size = grain_size
        self.num_bits = num_bits
        self.M2D = M2D
        self.save_path = save_path
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1.0 / self.half_lvls]), requires_grad=False)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  #in-place change MSB to negative

    def forward(self, input):
        if self.inf_with_weight:
            weight_rec = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, save_path = self.save_path)(self.weight * self.step_size)
            return F.conv2d(input, weight_rec * self.step_size, self.bias,
                            self.stride, self.padding, self.dilation,
                            self.groups)
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls) * self.step_size
            weight_rec = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, save_path = self.save_path)(weight_quan)* self.step_size
            return F.conv2d(input, weight_rec, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = torch.tensor(1.0) / self.half_lvls
            # self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size, self.half_lvls)
            # self.weight.data = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, save_path = self.save_path)(weight_quan)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True


class quan_Linear(nn.Linear):
    def __init__(self, in_features, out_features, grain_size = (1,1), num_bits = 3, M2D = 0.0, save_path = './'):
        super(quan_Linear, self).__init__(in_features, out_features, bias=True)
        self.N_bits = 8
        self.grain_size = grain_size
        self.num_bits = num_bits
        self.M2D = M2D
        self.save_path = save_path
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1.0 / self.half_lvls]), requires_grad=False)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  #in-place reverse

    def forward(self, input):
        if self.inf_with_weight:
            weight_rec = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, save_path = self.save_path)(self.weight* self.step_size)
            return F.linear(input, weight_rec * self.step_size, self.bias)
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls) * self.step_size
            weight_rec = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, save_path = self.save_path)(weight_quan)* self.step_size
            return F.linear(input, weight_rec, self.bias)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = torch.tensor(1.0) / self.half_lvls
            # self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size, self.half_lvls)
            # self.weight.data = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, save_path = self.save_path)(weight_quan)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True

