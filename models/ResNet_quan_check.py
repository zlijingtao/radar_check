import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from torch.nn import init
from .fixpoint_modules import  get_centroid, quantize
from .check_modules import get_qcode, get_code_2, get_rescue
from os import path
import config
# import time
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



class Unite(torch.autograd.Function):
    def __init__(self, grain_size, num_bits, M2D, step_size, half_lvls, save_path, check_gsize, check_factor, check_bit, codebook):
        super(Unite,self).__init__()
        self.grain_size = grain_size #grain size in tuple
        self.M2D = M2D
        self.num_bits = num_bits
        self.step_size = step_size
        self.half_lvls = half_lvls
        self.save_path = save_path
        self.checking_group = check_gsize
        self.checking_factor_sigma = check_factor
        self.checking_bit = check_bit
        self.codebook = codebook.cuda()
    def forward(self, input):
        self.save_for_backward(input)
        input = input/self.step_size
        self.centroid = get_centroid(input, self.grain_size, self.num_bits, self.M2D, self.half_lvls)
        global num_res
        global ti
        global ti2
        if input.device.index == 0:
            ti += 1
            if ti > num_res:
                ti = 1
                config.check = 0
        elif input.device.index == 1:
            ti2 += 1
            if ti2 > num_res:
                ti2 = 1
                config.check = 0
        input_d = (input - self.centroid)
        '''Internal Checker Setting'''
        transform = False
        rescue = True
        # rescue = False
        if input.device.index == 0:
            '''Run Checker'''
            if transform:
                input_d_transformed = input_d - 25 * (input_d > 0).float() * 1 + 25 * (input_d <= 0).float() * 1
                signature = get_qcode(input_d_transformed, [1,self.checking_group], self.checking_bit, self.half_lvls, factor = self.checking_factor_sigma, codebook = self.codebook)
                if not(path.exists(self.save_path + '/saved_tensors/signature{}.pt'.format(ti))):
                    if ti <=num_res:
                        mean_data = get_code_2(input_d_transformed, [1,self.checking_group], self.codebook).cpu().numpy()
                        # writer.add_histogram('mean_hist', mean_data, ti)
                        # writer.add_histogram('input_d_transformed_hist', input_d_transformed, ti)
                        torch.save(signature, self.save_path + '/saved_tensors/signature{}.pt'.format(ti))
                        signature_data = signature.cpu().numpy()
                        # writer.add_histogram('signature_hist', signature_data, ti)
                else:
                    orig_signature = torch.load(self.save_path + '/saved_tensors/signature{}.pt'.format(ti))
                    inequal = (signature - orig_signature != 0)
                    detected_item = torch.nonzero(inequal[::2] + inequal[1::2]).size(0)
                    if detected_item != 0:
                        config.check += detected_item
                        if rescue:
                            input_d = get_rescue(input_d, [1,self.checking_group], (signature - orig_signature == 0))
            else:
                signature = get_qcode(input_d, [1,self.checking_group], self.checking_bit, self.half_lvls, factor = self.checking_factor_sigma, codebook = self.codebook)
                # print(ti)
                # print(signature.size())
                if not(path.exists(self.save_path + '/saved_tensors/signature{}.pt'.format(ti))):
                    if ti <=num_res:
                        mean_data = get_code_2(input_d, [1,self.checking_group], self.codebook).cpu().numpy()
                        writer.add_histogram('mean_hist', mean_data, ti)
                        writer.add_histogram('input_d_hist', input_d, ti)
                        torch.save(signature, self.save_path + '/saved_tensors/signature{}.pt'.format(ti))
                        signature_data = signature.cpu().numpy()
                        writer.add_histogram('signature_hist', signature_data, ti)
                else:
                    orig_signature = torch.load(self.save_path + '/saved_tensors/signature{}.pt'.format(ti))
                    inequal = (signature - orig_signature != 0)
                    detected_item = torch.nonzero(inequal[::2] + inequal[1::2]).size(0)
                    if detected_item != 0:
                        config.check += detected_item
                        if rescue:
                            input_d = get_rescue(input_d, [1,self.checking_group], (signature - orig_signature == 0), self.checking_bit)
        elif input.device.index == 1:
            '''Run Checker'''
            if transform:
                input_d_transformed = input_d - 25 * (input_d > 0).float() * 1 + 25 * (input_d <= 0).float() * 1
                signature = get_qcode(input_d_transformed, [1,self.checking_group], self.checking_bit, self.half_lvls, factor = self.checking_factor_sigma, codebook = self.codebook)
                if not(path.exists(self.save_path + '/saved_tensors/signature{}.pt'.format(ti2))):
                    if ti2 <=num_res:
                        mean_data = get_code_2(input_d_transformed, [1,self.checking_group], self.codebook).cpu().numpy()
                        writer.add_histogram('mean_hist', mean_data, ti2)
                        writer.add_histogram('input_d_transformed_hist', input_d_transformed, ti2)
                        torch.save(signature, self.save_path + '/saved_tensors/signature{}.pt'.format(ti2))
                        signature_data = signature.cpu().numpy()
                        writer.add_histogram('signature_hist', signature_data, ti2)
                else:
                    orig_signature = torch.load(self.save_path + '/saved_tensors/signature{}.pt'.format(ti2))
                    inequal = (signature - orig_signature != 0)
                    detected_item = torch.nonzero(inequal[::2] + inequal[1::2]).size(0)
                    if detected_item != 0:
                        config.check += detected_item
                        if rescue:
                            input_d = get_rescue(input_d, [1,self.checking_group], (signature - orig_signature == 0))
            else:
                signature = get_qcode(input_d, [1,self.checking_group], self.checking_bit, self.half_lvls, factor = self.checking_factor_sigma, codebook = self.codebook)
                # print(ti2)
                # print(signature.size())
                if not(path.exists(self.save_path + '/saved_tensors/signature{}.pt'.format(ti2))):
                    if ti2 <=num_res:
                        mean_data = get_code_2(input_d, [1,self.checking_group], self.codebook).cpu().numpy()
                        writer.add_histogram('mean_hist', mean_data, ti2)
                        writer.add_histogram('input_d_hist', input_d, ti2)
                        torch.save(signature, self.save_path + '/saved_tensors/signature{}.pt'.format(ti2))
                        signature_data = signature.cpu().numpy()
                        writer.add_histogram('signature_hist', signature_data, ti2)
                else:
                    orig_signature = torch.load(self.save_path + '/saved_tensors/signature{}.pt'.format(ti2))
                    inequal = (signature - orig_signature != 0)
                    detected_item = torch.nonzero(inequal[::2] + inequal[1::2]).size(0)
                    if detected_item != 0:
                        config.check += detected_item
                        if rescue:
                            input_d = get_rescue(input_d, [1,self.checking_group], (signature - orig_signature == 0), self.checking_bit)
        output = input.clone().zero_()
        self.W = (1-self.M2D) * self.half_lvls
        deviation = input_d.clamp_(-self.W, self.W).round_()
        output = deviation + self.centroid

        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class quan_Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 codebook,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True, 
                 grain_size = (1,1), num_bits = 3, M2D = 0.0, save_path = './', check_gsize = 4, check_factor = 4.0, check_bit = 2):
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
        self.codebook = codebook
        self.check_gsize = check_gsize
        self.check_factor = check_factor
        self.check_bit = check_bit
        self.save_path = save_path
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = torch.Tensor([1.0 / self.half_lvls])
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = 2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float().cuda()

        self.b_w[0] = -self.b_w[0]  #in-place change MSB to negative

    def forward(self, input):
        if self.inf_with_weight:
            weight_rec = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, save_path = self.save_path,
                            check_gsize = self.check_gsize, check_factor = self.check_factor, check_bit = self.check_bit, codebook = self.codebook)(self.weight * self.step_size)
            return F.conv2d(input, weight_rec * self.step_size, self.bias,
                            self.stride, self.padding, self.dilation,
                            self.groups)
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls) * self.step_size
            weight_rec = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, save_path = self.save_path,
                            check_gsize = self.check_gsize, check_factor = self.check_factor, check_bit = self.check_bit, codebook = self.codebook)(weight_quan)* self.step_size
            return F.conv2d(input, weight_rec, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size = torch.tensor(1.0) / self.half_lvls
            # self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size,
                                        self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True


class quan_Linear(nn.Linear):
    def __init__(self, in_features, out_features, codebook, grain_size = (1,1), num_bits = 3, M2D = 0.0, save_path = './', check_gsize = 4, check_factor = 4.0, check_bit = 2):
        super(quan_Linear, self).__init__(in_features, out_features, bias=True)
        self.N_bits = 8
        self.grain_size = grain_size
        self.num_bits = num_bits
        self.M2D = M2D
        self.codebook = codebook
        self.check_gsize = check_gsize
        self.check_factor = check_factor
        self.check_bit = check_bit
        self.save_path = save_path
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = torch.Tensor([1.0 / self.half_lvls])
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = 2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float().cuda()

        self.b_w[0] = -self.b_w[0]  #in-place reverse

    def forward(self, input):
        if self.inf_with_weight:
            weight_rec = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, save_path = self.save_path,
                            check_gsize = self.check_gsize, check_factor = self.check_factor, check_bit = self.check_bit, codebook = self.codebook)(self.weight * self.step_size)
            return F.linear(input, weight_rec * self.step_size, self.bias)
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls) * self.step_size
            weight_rec = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, save_path = self.save_path,
                            check_gsize = self.check_gsize, check_factor = self.check_factor, check_bit = self.check_bit, codebook = self.codebook)(weight_quan)* self.step_size
            return F.linear(input, weight_rec, self.bias)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size = torch.tensor(1.0) / self.half_lvls
            # self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size,
                                        self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True

def conv3x3(in_planes, out_planes, codebook, stride=1, grain_size = (1,1), num_bits = 4, M2D = 0, save_path = './', check_gsize = 4, check_factor = 4.0, check_bit = 2):
    """3x3 convolution with padding"""
    return quan_Conv2d(in_planes, out_planes, kernel_size=3, codebook=codebook, stride=stride,
                     padding=1, bias=False, grain_size = grain_size, num_bits = num_bits, M2D = M2D, save_path = save_path, check_gsize = check_gsize, check_factor = check_factor, check_bit = check_bit)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, codebook, stride=1, downsample=None, grain_size = (1,1), num_bits = 4, M2D = 0, save_path = './', check_gsize = 4, check_factor = 4.0, check_bit = 2):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, codebook, stride, grain_size, num_bits, M2D, save_path, check_gsize, check_factor, check_bit)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, codebook, 1, grain_size, num_bits, M2D, save_path, check_gsize, check_factor, check_bit)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        # self.expansion = 1
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, codebook, stride=1, downsample=None, grain_size = (1,1), num_bits = 4, M2D = 0, save_path = './', check_gsize = 4, check_factor = 4.0, check_bit = 2):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1 = quan_Conv2d(inplanes, planes, kernel_size=1, codebook = codebook, stride=1, 
                                padding=0, bias=False, grain_size = grain_size, num_bits = num_bits, M2D = M2D, save_path = save_path, check_gsize = check_gsize, check_factor = check_factor, check_bit = check_bit)

        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        self.conv2 = quan_Conv2d(planes, planes, kernel_size=3, codebook = codebook, stride=stride, 
                                padding=1, bias=False, grain_size = grain_size, num_bits = num_bits, M2D = M2D, save_path = save_path, check_gsize = check_gsize, check_factor = check_factor, check_bit = check_bit)

        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.conv3 = quan_Conv2d(planes, planes * self.expansion, kernel_size=1, codebook = codebook, stride=1, 
                                padding=0, bias=False, grain_size = grain_size, num_bits = num_bits, M2D = M2D, save_path = save_path, check_gsize = check_gsize, check_factor = check_factor, check_bit = check_bit)
        
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, codebook, fp_fl=True, fp_ll=True, 
  input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, 
  res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, 
  output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0,
  save_path = './', check_gsize = 4, check_factor = 4.0, check_bit = 2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if fp_fl:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = quan_Conv2d(3, 64, kernel_size=7, codebook = codebook, stride=2, padding=3, bias=False, grain_size = input_grain_size, num_bits = input_num_bits, M2D = input_M2D, save_path = save_path, check_gsize = check_gsize, check_factor = check_factor, check_bit = check_bit)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], codebook, 1, res_grain_size, res_num_bits ,res_M2D, save_path = save_path, check_gsize = check_gsize, check_factor = check_factor, check_bit = check_bit)
        self.layer2 = self._make_layer(block, 128, layers[1], codebook, 2, res_grain_size, res_num_bits ,res_M2D, save_path = save_path, check_gsize = check_gsize, check_factor = check_factor, check_bit = check_bit)
        self.layer3 = self._make_layer(block, 256, layers[2], codebook, 2, res_grain_size, res_num_bits ,res_M2D, save_path = save_path, check_gsize = check_gsize, check_factor = check_factor, check_bit = check_bit)
        self.layer4 = self._make_layer(block, 512, layers[3], codebook, 2, res_grain_size, res_num_bits ,res_M2D, save_path = save_path, check_gsize = check_gsize, check_factor = check_factor, check_bit = check_bit)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if fp_ll:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = quan_Linear(512 * block.expansion, num_classes, codebook, output_grain_size, output_num_bits, output_M2D, save_path = save_path, check_gsize = check_gsize, check_factor = check_factor, check_bit = check_bit)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, codebook, stride=1, grain_size = (1,1), num_bits = 4, M2D = 0, save_path = './', check_gsize = 4, check_factor = 4.0, check_bit = 2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                quan_Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, codebook = codebook, stride=stride, padding = 0, bias=False, grain_size = grain_size, num_bits = num_bits, M2D = M2D, save_path = save_path, check_gsize = check_gsize, check_factor = check_factor, check_bit = check_bit),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, codebook, stride, downsample, grain_size, num_bits, M2D, save_path, check_gsize, check_factor, check_bit))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, codebook, 1, None, grain_size, num_bits, M2D, save_path, check_gsize, check_factor, check_bit))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18b_quan_c(codebook, num_classes=1000, input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0, save_path = './', check_gsize = 4, check_factor = 4.0, check_bit = 2):
    global ti
    global ti2
    ti = 0
    ti2 = 0
    global num_res
    num_res = 21
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, codebook, False, False, input_grain_size, input_num_bits, input_M2D, res_grain_size, res_num_bits, res_M2D, output_grain_size, output_num_bits, output_M2D, save_path, check_gsize, check_factor, check_bit)
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34b_quan_c(codebook, num_classes=1000, input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0, save_path = './', check_gsize = 4, check_factor = 4.0, check_bit = 2):
    global ti
    global ti2
    ti = 0
    ti2 = 0
    global num_res
    num_res = 36
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, codebook, False, False, input_grain_size, input_num_bits, input_M2D, res_grain_size, res_num_bits, res_M2D, output_grain_size, output_num_bits, output_M2D, save_path, check_gsize, check_factor, check_bit)
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet50b_quan_c(codebook, num_classes=1000, input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0, save_path = './', check_gsize = 4, check_factor = 4.0, check_bit = 2):
    global ti
    global ti2
    ti = 0
    ti2 = 0
    global num_res
    num_res = 52
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, codebook, False, False, input_grain_size, input_num_bits, input_M2D, res_grain_size, res_num_bits, res_M2D, output_grain_size, output_num_bits, output_M2D, save_path, check_gsize, check_factor, check_bit)
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101b_quan_c(codebook, num_classes=1000, input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0, save_path = './', check_gsize = 4, check_factor = 4.0, check_bit = 2):
    global ti
    global ti2
    ti = 0
    ti2 = 0
    global num_res
    num_res = 103
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, codebook, False, False, input_grain_size, input_num_bits, input_M2D, res_grain_size, res_num_bits, res_M2D, output_grain_size, output_num_bits, output_M2D, save_path, check_gsize, check_factor, check_bit)
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model
