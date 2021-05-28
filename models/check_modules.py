import torch
import torch.nn.functional as F
import math
import numpy as np

def get_rescue(input, grain_size, rescue_mask, num_bits=2):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if num_bits == 2:
        rescue_mask = rescue_mask[::2] & rescue_mask[1::2]
    else:
        rescue_mask = rescue_mask[::1]
    # print(rescue_mask)
    rescue_mask = rescue_mask.view(-1,1).repeat(1,grain_size[1]).cuda().float()
    if len(input.size()) == 2:
        original_size = input.size()
        #Add obfuscation (deinterleave)
        
        input = shuffle_weight(input)
        
        if (input.size()[0]*input.size()[1])% grain_size[1] == 0:
            reshaped_input = input.contiguous().view(-1, grain_size[1])
            output = torch.mul(rescue_mask, reshaped_input)
            output = output.view(original_size[0], original_size[1])
        else:
            correct_shape = int(np.floor(input.size()[0]*input.size()[1] / grain_size[1]))*grain_size[1]
            flattened_input = input.contiguous().flatten()
            reshaped_input_rest = flattened_input[correct_shape:]
            if correct_shape != 0:
                reshaped_input = flattened_input[:correct_shape].view(-1, grain_size[1])
                rescue_mask_1 = rescue_mask[:-1, :]
                output = torch.mul(rescue_mask_1, reshaped_input)
                flattened_output = output.flatten()
                output_rest = torch.mul(reshaped_input_rest, rescue_mask_1[-1,:reshaped_input_rest.size()[0]]).flatten()
                output = torch.cat((flattened_output, output_rest))
            else:
                output = torch.mul(reshaped_input_rest, rescue_mask[-1,:reshaped_input_rest.size()[0]]).flatten()
            output = output.view(original_size[0], original_size[1])
        # Add obfuscation (deinterleave)
        # output = output.t()
        output = shuffle_weight_back(output)
        
    if len(input.size()) == 4:
        initial_size = input.size()
        input = input.contiguous().permute(1, 2, 3, 0).view(-1, initial_size[0])
        #Add obfuscation (deinterleave)
        
        input = shuffle_weight(input)
        
        original_size = input.size()
        if (input.size()[0]*input.size()[1])% grain_size[1] == 0:
            reshaped_input = input.contiguous().view(-1, grain_size[1])
            output = torch.mul(rescue_mask, reshaped_input)
            
            output = output.view(original_size[0], original_size[1])
        else:
            correct_shape = int(np.floor(input.size()[0]*input.size()[1] / grain_size[1]))*grain_size[1]
            flattened_input = input.contiguous().flatten()
            reshaped_input_rest = flattened_input[correct_shape:]
            if correct_shape != 0:
                reshaped_input = flattened_input[:correct_shape].view(-1, grain_size[1])
                # print(input.size())
                # print(reshaped_input.size())
                # print(rescue_mask.size())
                rescue_mask_1 = rescue_mask[:-1, :]
                
                output = torch.mul(rescue_mask_1, reshaped_input)
                flattened_output = output.flatten()
                output_rest = torch.mul(reshaped_input_rest, rescue_mask[-1,:reshaped_input_rest.size()[0]]).flatten()
                # print(flattened_output.size())
                # print(output_rest.size())
                output = torch.cat((flattened_output, output_rest))
            else:
                output = torch.mul(reshaped_input_rest, rescue_mask[-1,:reshaped_input_rest.size()[0]]).flatten()
            output = output.view(original_size[0], original_size[1])
        # Add obfuscation (deinterleave)
        # output = output.t()
        output = shuffle_weight_back(output)
        
        output = output.view(initial_size[1], initial_size[2], initial_size[3], initial_size[0]).permute(3, 0, 1, 2)
    return output

def get_qcode(input, grain_size, num_bits, half_lvls, factor, codebook):
    if len(input.size()) == 2:
        original_size = input.size()
        reshaped_input = input.view(original_size[0], original_size[1])
        reshaped_input = shuffle_weight(reshaped_input)
        pooling_result = get_code(reshaped_input, grain_size, codebook)
        output = get_mean_quantized_int8(pooling_result, num_bits)
    if len(input.size()) == 4:
        original_size = input.size()
        reshaped_input = input.permute(1, 2, 3, 0).view(-1, original_size[0])
        reshaped_input = shuffle_weight(reshaped_input)
        pooling_result = get_code(reshaped_input, grain_size, codebook)
        output = get_mean_quantized_int8(pooling_result, num_bits)
    return output
    
def shuffle_weight(input):
    #input needs to be 2D array
    input = input.cpu().detach().numpy()
    original_size = input.shape
    input = input.flatten()
    # print(input)
    np.random.seed(25)
    # perm = permutation(len(input), 4)
    perm = np.random.permutation(len(input))
    input = input[perm]
    input = input.reshape(original_size)
    # np.take(input,np.random.permutation(input.shape[0]),axis=0,out=input)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.from_numpy(input).float().to(device)
    return input
# def permutation(nb_rows, nb_frames):
    # return [frame*nb_rows//nb_frames + row for row in range(nb_rows // nb_frames)      
                                           # for frame in range(nb_frames)]
def shuffle_weight_back(input):
    #input needs to be 2D array
    input = input.cpu().detach().numpy()
    original_size = input.shape
    input = input.flatten()
    # print(input)
    np.random.seed(25)
    perm = np.random.permutation(len(input))
    perm_b = invert_permutation(perm)
    input = input[perm_b]
    input = input.reshape(original_size)
    # np.take(input,np.random.permutation(input.shape[0]),axis=0,out=input)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.from_numpy(input).float().to(device)
    return input
    
def invert_permutation(p):
    '''The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    Returns an array s, where s[i] gives the index of i in p.
    '''
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s
    
def get_code(reshaped_input, grain_size, codebook):
    assert grain_size[1] == codebook.size()[1] , 'codebook and grain size do not match!'
    reshaped_input = reshaped_input.char()
    if reshaped_input.size()[0]*reshaped_input.size()[1] % grain_size[1] == 0:
        reshaped_input = reshaped_input.contiguous().view(-1, grain_size[1])
        repeat_time = math.floor(reshaped_input.size()[0]/codebook.size()[0])
        rest = reshaped_input.size()[0] - repeat_time*codebook.size()[0]
        rest_handle = 0
    else:
        correct_shape = int(np.floor(reshaped_input.size()[0]*reshaped_input.size()[1] / grain_size[1]))*grain_size[1]
        flattened_input = reshaped_input.contiguous().flatten()
        if correct_shape == 0:
            output, overflow = get_sum(torch.mul(flattened_input, codebook[0,:flattened_input.size()[0]]))
            return torch.cat((output.view(1), overflow.view(1))).reshape(2,-1)
        reshaped_input_rest = flattened_input[correct_shape:]
        reshaped_input = flattened_input[:correct_shape].view(-1, grain_size[1])
        repeat_time = math.floor(reshaped_input.size()[0]/codebook.size()[0])
        rest = reshaped_input.size()[0] - repeat_time*codebook.size()[0]
        rest_handle = 1
    
    if repeat_time !=0:
        codebook_bc = codebook.repeat(repeat_time, 1)
        for i in range(rest):
            codebook_bc = torch.cat((codebook_bc, codebook[i,:].view(1,-1)), 0)
        assert reshaped_input.size() == codebook_bc.size() , 'Debug'
        reshaped_input = reshaped_input.cuda()
        codebook_bc = codebook_bc.cuda()
        output = torch.mul(reshaped_input, codebook_bc)
        output, overflow = get_sum(output)
        if rest_handle == 1:
            coded_rest, overflow_rest = get_sum(torch.mul(reshaped_input_rest, codebook_bc[0,:reshaped_input_rest.size()[0]]))
            output = torch.cat((output, coded_rest.view(1)))
            overflow = torch.cat((overflow, overflow_rest.view(1)))
    else:
        output, overflow = get_sum(torch.mul(reshaped_input, codebook[:reshaped_input.size()[0],:]))
        if rest_handle == 1:
            coded_rest, overflow_rest = get_sum(torch.mul(reshaped_input_rest, codebook[reshaped_input.size()[0],:reshaped_input_rest.size()[0]]))
            output = torch.cat((output, coded_rest.view(1)))
            overflow = torch.cat((overflow, overflow_rest.view(1)))
    return torch.cat((output, overflow), dim = 0).reshape(2,-1)

def get_sum(input):
    if len(input.size()) == 2:
        output = torch.sum(input, dim = 1, dtype = torch.int8)
        overflow = torch.sum((input < 0), dim = 1, dtype = torch.int8)
    else:
        output = torch.sum(input, dtype = torch.int8)
        overflow = torch.sum((input < 0), dtype = torch.int8)
    return output, overflow
def get_code_2(input, grain_size, codebook):
    if len(input.size()) == 2:
        original_size = input.size()
        reshaped_input = input.view(original_size[0], original_size[1])
        pooling_result = get_code(reshaped_input, grain_size, codebook)

    if len(input.size()) == 4:
        original_size = input.size()
        reshaped_input = input.permute(1, 2, 3, 0).view(-1, original_size[0])
        pooling_result = get_code(reshaped_input, grain_size, codebook)
    return pooling_result

def get_mean(input, grain_size):
    if len(input.size()) == 2:
        original_size = input.size()
        reshaped_input = input.view(1, 1, original_size[0], original_size[1])
        pooling_result = F.avg_pool2d(reshaped_input, grain_size, grain_size)
        # print(pooling_result.type())
    if len(input.size()) == 4:
        original_size = input.size()
        reshaped_input = input.permute(1, 2, 3, 0).view(1, 1, -1, original_size[0])
        pooling_result = F.avg_pool2d(reshaped_input, grain_size, grain_size)
    return pooling_result

def get_mean_bits(input, grain_size, num_bits, half_lvls, factor):
    if len(input.size()) == 2:
        original_size = input.size()
        reshaped_input = input.view(1, 1, original_size[0], original_size[1])
        pooling_result = F.avg_pool2d(reshaped_input, grain_size, grain_size)
        # print(pooling_result.type())
        output = get_mean_quantized(pooling_result, num_bits, half_lvls, factor)
        # pooling_result = pooling_result.view(pooling_result.size()[2:])
        # pooling_result = pooling_result.unsqueeze(1).repeat(1,grain_size[0], 1).view(-1,pooling_result.size()[1]).transpose(0,1)
        # output = pooling_result.repeat(1, grain_size[1]).view(-1,pooling_result.size()[1]).transpose(0,1)
    if len(input.size()) == 4:
        original_size = input.size()
        reshaped_input = input.permute(1, 2, 3, 0).view(1, 1, -1, original_size[0])
        pooling_result = F.avg_pool2d(reshaped_input, grain_size, grain_size)
        output = get_mean_quantized(pooling_result, num_bits, half_lvls, factor)
        # pooling_result = pooling_result.view(pooling_result.size()[2:])
        # pooling_result = pooling_result.unsqueeze(1).repeat(1,grain_size[0], 1).view(-1,pooling_result.size()[1]).transpose(0,1)
        # pooling_result = pooling_result.repeat(1, grain_size[1]).view(-1,pooling_result.size()[1]).transpose(0,1)
        # output = pooling_result.view(original_size[1], original_size[2], original_size[3], original_size[0]).permute(3, 0, 1, 2)
    return output
    
def get_mean_quantized(input, num_bits, half_lvls, factor):
    output = input.clone()
    qmin = 0
    qmax = qmin + 2.**num_bits - 1.
    '''full DR'''
    # scale = (torch.max(output) - torch.min(output))*1.0/ (qmax - qmin)
    # print((torch.max(output) - torch.min(output))*0.9)
    '''sigma DR'''
    scale = output.std() * factor / (qmax - qmin)
    # print(output.std() * factor)
    output.div_(scale)
    output.add_((qmax - qmin)/2)
    output.clamp_(qmin, qmax).round_()
    
    # output.add_(-(qmax - qmin)/2)
    # output.mul_(scale).round_()
    # print(np.unique(output.cpu().numpy()))
    output = output.type(torch.uint8)
    # print(np.unique(output.cpu().numpy()))
    # print(output.type())
    return output
    
def get_mean_quantized_int8(input, num_bits):

    output = input.clone().cpu().numpy()
    overflow = output[1, :]
    # print(overflow)
    overflow_bit = (((overflow % 4) > 1) ).view(np.uint8).reshape(-1, 1)
    # print(overflow_bit)
    output = output[0, :]
    output = output.astype(np.int8)
    output = output.view(np.uint8)
    output += 128
    output = output.reshape(-1,1)
    output_bit = np.unpackbits(output, axis=1).reshape(-1, 8)
    '''one-bit parity/two-bit weighted sum'''
    if num_bits == 1:
        mixed_bit = output_bit
    else:
        mixed_bit = np.concatenate((overflow_bit, output_bit), axis = 1)
    
    output_concate = mixed_bit[:, :num_bits].flatten()
    # output_concate = output_bit[:, :num_bits].flatten()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_torch = torch.from_numpy(output_concate).to(device)
    return output_torch