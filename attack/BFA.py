import torch
import operator
from attack.data_conversion import *
import models
import numpy as np
from tensorboardX import SummaryWriter
import config
import random

def Get_grain_topk(grad, k_top, flip_allowed, grain_size):
    f_grad = grad
    grain_sizes = grain_size[0]*grain_size[1]
    num_grain = int(list(f_grad.size())[0]/grain_sizes)
    
    num_of_flip_allowed = 1
    if k_top >= num_of_flip_allowed * num_grain:
        print("k_top is not in allowed range")
    grad_topk = []
    idx_topk = []
    for i in range(num_grain):
        current_grain = f_grad[i*grain_sizes:(i+1)*grain_sizes]
        grain_topk, grain_idx_topk = current_grain.topk(num_of_flip_allowed)
        grain_idx_topk += i*grain_sizes
        grad_topk.extend(list(grain_topk))
        idx_topk.extend(list(grain_idx_topk))
    grad_topk = torch.stack(grad_topk)
    idx_topk = torch.stack(idx_topk)
    
    grad_topk, new_idx = grad_topk.topk(k_top)
    idx_topk = idx_topk[new_idx]
    return grad_topk, idx_topk


def Get_random_topk(grad, k_top):
    f_grad = grad
    perm = torch.randperm(f_grad.size(0))
    idx = perm[:k_top]
    grad_r = f_grad[idx]
    return grad_r, idx

def update_grain_weight_mask(grain_mask, flip_allowed, grain_size, target_idx):
    grain_sizes = grain_size[0]*grain_size[1]
    original_size = grain_mask.size()
    if len(original_size) == 4:
        grain_mask_item = grain_mask.permute(1, 2, 3, 0).contiguous().view(-1)
    else:
        grain_mask_item = grain_mask.contiguous().view(-1)
    num_grain = int(list(grain_mask_item.size())[0]/grain_sizes)
    # mark the target to 0
    grain_mask_item[target_idx] = 0
    # If multiple weights (depends on flip_allowed) in a grain has already set to zero, then set the whole grain mask to zero
    for i in range(num_grain):
        current_grain = grain_mask_item[i*grain_sizes:(i+1)*grain_sizes]
        if int(list(current_grain.nonzero().view(-1).size())[0]) < (grain_sizes + 1 - flip_allowed):
            grain_mask_item[i*grain_sizes:(i+1)*grain_sizes] = 0
    if len(original_size) == 4:
        grain_mask_orig = grain_mask_item.contiguous().view(original_size[1], original_size[2], original_size[3], original_size[0]).permute(3, 0, 1, 2)
    else:
        grain_mask_orig = grain_mask_item.contiguous().view(original_size)
    # print(grain_mask_orig)
    return grain_mask_orig

    
class BFA(object):
    def __init__(self, criterion, k_top=10, flip_allowed = 1, layer_id = 0):

        self.criterion = criterion
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        self.grain_mask = {}
        self.attack_bit = {}
        self.attack_weight_value = dict.fromkeys(['0_32', '32_64', '64_96', '96_128', '128_160','160_192','192_224','224_256'], 0)
        self.attack_index = {}
        self.attack_layer = {}
        self.bit_counter = 0
        self.k_top = k_top
        self.n_bits2flip = 0
        self.loss = 0
        self.layer_id = layer_id
        self.flip_allowed = k_top
        if flip_allowed <= k_top:
            self.flip_allowed = flip_allowed
    def flip_bit(self, name, m, update_mask = False, stats = False):
        '''
        the data type of input param is 32-bit floating, then return the data should
        be in the same data_type.
        '''
        self.k_top = m.weight.detach().flatten().__len__()
        # a. use the grain_mask to mask current gradient. so unflippable location always has gradient 0.
            # writer = SummaryWriter('runs/grad_map_1/grad_data_0')
            # grad_map = (m.weight.grad.detach().abs().cuda()
            # writer.add_histogram('weight_hist',grad_map, 0)
        if len(m.weight.grad.detach().size()) == 4:
            masked_grad = (m.weight.grad.detach().abs().cuda()* self.grain_mask[name].cuda()).permute(1, 2, 3, 0).contiguous().view(-1) 
        else:
            masked_grad = (m.weight.grad.detach().abs().cuda() * self.grain_mask[name].cuda()).contiguous().view(-1)
        # 1. flatten the gradient tensor to perform topk (only count the grain wise biggest)
        # w_grad_topk, w_idx_topk = Get_grain_topk(masked_grad, self.k_top, self.flip_allowed, m.grain_size)
        w_grad_topk, w_idx_topk = masked_grad.topk(self.k_top)
        # print(w_grad_topk)
        # print('========')
        # update the b_grad to its signed representation
        if len(m.weight.grad.detach().size()) == 4:
            w_grad_topk = m.weight.grad.detach().permute(1, 2, 3, 0).contiguous().view(-1)[w_idx_topk]
            # print(w_grad_topk)
        else:
            w_grad_topk = m.weight.grad.detach().contiguous().view(-1)[w_idx_topk]

        # 2. create the b_grad matrix in shape of [N_bits, k_top]
        b_grad_topk = w_grad_topk * m.b_w.data

        # 3. generate the gradient mask to zero-out the bit-gradient
        # which can not be flipped
        b_grad_topk_sign = (b_grad_topk.sign() +
                            1) * 0.5  # zero -> negative, one -> positive
        # convert to twos complement into unsigned integer
        if len(m.weight.grad.detach().size()) == 4:
            # print(m.weight.detach())
            w_bin = int2bin(m.weight.detach().permute(1, 2, 3, 0).contiguous().view(-1), m.N_bits).short()
            # print(w_bin)
        else:
            w_bin = int2bin(m.weight.detach().contiguous().view(-1), m.N_bits).short()
        w_bin_topk = w_bin[w_idx_topk]  # get the weights whose grads are topk
        
        # generate two's complement bit-map
        b_bin_topk = (w_bin_topk.repeat(m.N_bits,1) & m.b_w.abs().repeat(1,self.k_top).short()) \
        // m.b_w.abs().repeat(1,self.k_top).short()
        grad_mask = b_bin_topk ^ b_grad_topk_sign.short()
        # print(b_grad_topk_sign.short())


        # 4. apply the gradient mask upon ```b_grad_topk``` and in-place update it
        b_grad_topk *= grad_mask.float()

        # 5. identify the several maximum of absolute bit gradient and return the
        # index, the number of bits to flip is self.n_bits2flip
        grad_max = b_grad_topk.abs().max()
        _, b_grad_max_idx = b_grad_topk.abs().view(-1).topk(self.n_bits2flip)
        
        bit2flip = b_grad_topk.clone().view(-1).zero_()
        
        if grad_max.item() != 0:  # ensure the max grad is not zero
            bit2flip[b_grad_max_idx] = 1
            # print(bit2flip[b_grad_max_idx])
            # print(bit2flip.size())
            
            bit2flip = bit2flip.view(b_grad_topk.size())
        else:
            pass

        # print(bit2flip)
        # print(bit2flip.size())

        flipped_index = w_idx_topk[bit2flip.short().nonzero()[0][1]]
        if stats:
            attack_bit_position = torch.nonzero((bit2flip.short() * m.b_w.abs().short()).sum(0, dtype=torch.int16))[0]
            w_bin_topk_flipped = (bit2flip.short() * m.b_w.abs().short()).sum(0, dtype=torch.int16) \
                ^ w_bin_topk
            
            attack_weight_change = w_bin_topk_flipped[attack_bit_position.item()] - w_bin_topk[attack_bit_position.item()]
            if attack_weight_change.item() in self.attack_bit.keys():
                self.attack_bit[attack_weight_change.item()] += 1
            else:
                self.attack_bit[attack_weight_change.item()] = 1
                
            
            targeted_weight = w_bin_topk[attack_bit_position.item()].item()
            if targeted_weight< 128:
                if targeted_weight< 64:
                    if targeted_weight< 32:
                        self.attack_weight_value['0_32'] += 1
                    else:
                        self.attack_weight_value['32_64'] += 1
                else:
                    if targeted_weight< 96:
                        self.attack_weight_value['64_96'] += 1
                    else:
                        self.attack_weight_value['96_128'] += 1
            else:
                if targeted_weight> 192:
                    if targeted_weight> 224:
                        self.attack_weight_value['224_256'] += 1
                    else:
                        self.attack_weight_value['192_224'] += 1
                else:
                    if targeted_weight> 160:
                        self.attack_weight_value['160_192'] += 1
                    else:
                        self.attack_weight_value['128_160'] += 1
            # if (str(flipped_index.item()) + '_' + str(attack_weight_change.item())) in self.attack_index.keys():
                # self.attack_index[(str(flipped_index.item()) + '_' + str(attack_weight_change.item()))] += 1
            # else:
                # self.attack_index[(str(flipped_index.item()) + '_' + str(attack_weight_change.item()))] = 1
            # if (str(flipped_index.item()) + '_' + str(attack_weight_change.item())) in self.attack_index.keys():
                # self.attack_index[name + '_' + str(flipped_index.item())] += 1
            # else:
                # self.attack_index[name + '_' + str(flipped_index.item())] = 1
            
            if not name+'_G64' in self.attack_index.keys():
                self.attack_index[name+'_G64'] = {}
            if not name+'_G128' in self.attack_index.keys():
                self.attack_index[name+'_G128'] = {}
            if not name+'_G256' in self.attack_index.keys():
                self.attack_index[name+'_G256'] = {}
            if not name+'_G512' in self.attack_index.keys():
                self.attack_index[name+'_G512'] = {}
            if not name+'_G1024' in self.attack_index.keys():
                self.attack_index[name+'_G1024'] = {}
            if str(int(flipped_index.item()/64)) in self.attack_index[name+'_G64'].keys():
                self.attack_index[name+'_G64'][str(int(flipped_index.item()/64))] += 1
            else:
                self.attack_index[name+'_G64'][str(int(flipped_index.item()/64))] = 1
                
            if str(int(flipped_index.item()/128)) in self.attack_index[name+'_G128'].keys():
                self.attack_index[name+'_G128'][str(int(flipped_index.item()/128))] += 1
            else:
                self.attack_index[name+'_G128'][str(int(flipped_index.item()/128))] = 1
                
            if str(int(flipped_index.item()/256)) in self.attack_index[name+'_G256'].keys():
                self.attack_index[name+'_G256'][str(int(flipped_index.item()/256))] += 1
            else:
                self.attack_index[name+'_G256'][str(int(flipped_index.item()/256))] = 1
                
            if str(int(flipped_index.item()/512)) in self.attack_index[name+'_G512'].keys():
                self.attack_index[name+'_G512'][str(int(flipped_index.item()/512))] += 1
            else:
                self.attack_index[name+'_G512'][str(int(flipped_index.item()/512))] = 1
                
            if str(int(flipped_index.item()/1024)) in self.attack_index[name+'_G1024'].keys():
                self.attack_index[name+'_G1024'][str(int(flipped_index.item()/1024))] += 1
            else:
                self.attack_index[name+'_G1024'][str(int(flipped_index.item()/1024))] = 1
            
            
            if name in self.attack_layer.keys():
                self.attack_layer[name] += 1
            else:
                self.attack_layer[name] = 1
        # print(self.attack_bit)
        # 6. Based on the identified bit indexed by ```bit2flip```, generate another
        # mask, then perform the bitwise xor operation to realize the bit-flip.
        w_bin_topk_flipped = (bit2flip.short() * m.b_w.abs().short()).sum(0, dtype=torch.int16) \
                ^ w_bin_topk
               
        # b. If actually use this attack (on weight). update the grain mask, so that ones a weight flipped inside a grain, the grain will not be considered next time.
        if update_mask == True:
            self.grain_mask[name] = update_grain_weight_mask(self.grain_mask[name], self.flip_allowed, m.grain_size, flipped_index)
            # print(self.grain_mask[name])
        # 7. update the weight in the original weight tensor
        w_bin[w_idx_topk] = w_bin_topk_flipped  # in-place change
        if len(m.weight.grad.detach().size()) == 4:
            param_flipped = bin2int(w_bin, m.N_bits).contiguous().view(m.weight.data.size()[1], m.weight.data.size()[2], m.weight.data.size()[3], m.weight.data.size()[0]).permute(3, 0, 1, 2).float()
        else:
            param_flipped = bin2int(w_bin, m.N_bits).contiguous().view(m.weight.data.size()).float()
        
        return param_flipped, flipped_index, masked_grad[flipped_index-5:flipped_index+5]
        
    def progressive_bit_search(self, model, data, target, update_mask_flag = False, save_stats = False, twin = False, lessen = False):
        ''' 
        Given the model, base on the current given data and target, go through
        all the layer and identify the bits to be flipped. 
        '''
        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
        model.eval()
        gradient = {}
        # 1. perform the inference w.r.t given data and target
        output = model(data)
        self.loss = self.criterion(output, target)
        # 2. zero out the grads first, then get the grads
        for name, m in model.named_modules():
            if ("conv" in name) or ("classifier" in name) or ("fc" in name) or ("downsample.0" in name):
                if not(name in self.grain_mask):
                    self.grain_mask[name] = torch.ones(m.weight.data.detach().size())
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()
        self.loss.backward()
        # init the loss_max to enable the while loop
        self.loss_max = self.loss.item()
        # 3. for each layer flip #bits = self.bits2flip
        while self.loss_max <= self.loss.item():

            self.n_bits2flip += 1
            counter = 0
            for name, module in model.named_modules():
                if ("conv" in name) or ("classifier" in name) or ("fc" in name) or ("downsample.0" in name):
                    counter += 1
                    # clean_weight = module.weight.data.detach()
                    if counter == self.layer_id:
                        clean_weight = module.weight.data.detach()
                        
                        attack_weight, nouse_index, _ = self.flip_bit(name, module)
                        
                        module.weight.data = attack_weight
                        
                        # print("****************")
                        # module.weight.data = self.grain_mask[name]* (attack_weight - clean_weight) + clean_weight
                        
                        output = model(data)
                        # print(output)
                        gradient[name] = module.weight.grad.detach().abs()
                        self.loss_dict[name] = self.criterion(output, target).item()
                        
                        # change the weight back to the clean weight
                        module.weight.data = clean_weight
                    
                    elif self.layer_id == 0:
                        clean_weight = module.weight.data.detach()
                        
                        attack_weight, nouse_index, _ = self.flip_bit(name, module)
                        # print(attack_weight.size())
                        ###############

                        ################
                        module.weight.data = attack_weight
                        # print((reshaped_attack_weight - reshaped_clean_weight).nonzero())
                        # print("****************")
                        # module.weight.data = self.grain_mask[name]* (attack_weight - clean_weight) + clean_weight
                        
                        output = model(data)
                        # print(output)
                        gradient[name] = module.weight.grad.detach().abs()
                        self.loss_dict[name] = self.criterion(output, target).item()
                        
                        # change the weight back to the clean weight
                        module.weight.data = clean_weight
            # after going through all the layer, now we find the layer with max loss
            max_loss_module = max(self.loss_dict.items(),
                                  key=operator.itemgetter(1))[0]
            self.loss_max = self.loss_dict[max_loss_module]

        # 4. if the loss_max does lead to the degradation compared to the self.loss,
        # then change the that layer's weight without putting back the clean weight
        for name, module in model.named_modules():
            # print(name)
            if name == max_loss_module:
                # print(name, self.loss.item(), loss_max)
                # clean_weight = module.weight.data.detach()
                # print(clean_weight)
                clean_weight = module.weight.data.detach()
                attack_weight, attack_index, attack_gradient = self.flip_bit(name, module, update_mask = update_mask_flag, stats = True)
                attack_name = name
                if save_stats:
                    config.grad_map = gradient[name].cpu().numpy()
                    print("Attacking layer:" + name)
                ##############
                if twin == True:
                    if len(attack_weight.size()) == 4:
                        original_size = attack_weight.size()
                        reshaped_clean_weight = clean_weight.permute(1, 2, 3, 0)
                        reshaped_attack_weight = attack_weight.permute(1, 2, 3, 0)
                        initial_size = reshaped_attack_weight.size()
                        reshaped_clean_weight = reshaped_clean_weight.view(-1, original_size[0])
                        reshaped_attack_weight = reshaped_attack_weight.view(-1, original_size[0])
                        reshaped_clean_weight = reshaped_clean_weight.flatten()
                        reshaped_attack_weight = reshaped_attack_weight.flatten()
                    else:
                        original_size = attack_weight.size()
                        reshaped_clean_weight = clean_weight.view(original_size[0], original_size[1])
                        reshaped_attack_weight = attack_weight.view(original_size[0], original_size[1])
                        reshaped_clean_weight = reshaped_clean_weight.flatten()
                        reshaped_attack_weight = reshaped_attack_weight.flatten()
                    nonzero_index = (reshaped_attack_weight - reshaped_clean_weight).nonzero()[0]
                    print(nonzero_index)
                    neighbor_index = nonzero_index + 1
                    if nonzero_index.item() == 0:
                        neighbor_index = neighbor_index
                    elif (neighbor_index.item()/4) != (nonzero_index.item()/4):
                        neighbor_index = nonzero_index - 1
                    changes = (reshaped_attack_weight - reshaped_clean_weight)[nonzero_index]
                    if reshaped_attack_weight[neighbor_index] > 0:
                        reshaped_attack_weight[neighbor_index] -= 128
                    else:
                        reshaped_attack_weight[neighbor_index] += 128
                    if len(attack_weight.size()) == 4:
                        attack_weight = reshaped_attack_weight.view(-1, original_size[0]).view(initial_size).permute(3, 0, 1, 2)
                    else:
                        attack_weight = reshaped_attack_weight.view(original_size[0], original_size[1]).view(original_size)
                if lessen == True:
                    if len(attack_weight.size()) == 4:
                        original_size = attack_weight.size()
                        reshaped_clean_weight = clean_weight.permute(1, 2, 3, 0)
                        reshaped_attack_weight = attack_weight.permute(1, 2, 3, 0)
                        initial_size = reshaped_attack_weight.size()
                        reshaped_clean_weight = reshaped_clean_weight.view(-1, original_size[0])
                        reshaped_attack_weight = reshaped_attack_weight.view(-1, original_size[0])
                        reshaped_clean_weight = reshaped_clean_weight.flatten()
                        reshaped_attack_weight = reshaped_attack_weight.flatten()
                    else:
                        original_size = attack_weight.size()
                        reshaped_clean_weight = clean_weight.view(original_size[0], original_size[1])
                        reshaped_attack_weight = attack_weight.view(original_size[0], original_size[1])
                        reshaped_clean_weight = reshaped_clean_weight.flatten()
                        reshaped_attack_weight = reshaped_attack_weight.flatten()
                    nonzero_index = (reshaped_attack_weight - reshaped_clean_weight).nonzero()[0]
                    neighbor_index = nonzero_index
                    changes = (reshaped_attack_weight - reshaped_clean_weight)[nonzero_index]
                    if reshaped_attack_weight[neighbor_index] > 0:
                        reshaped_attack_weight[neighbor_index] -= 64
                    else:
                        reshaped_attack_weight[neighbor_index] += 64
                    if len(attack_weight.size()) == 4:
                        attack_weight = reshaped_attack_weight.view(-1, original_size[0]).view(initial_size).permute(3, 0, 1, 2)
                    else:
                        attack_weight = reshaped_attack_weight.view(original_size[0], original_size[1]).view(original_size)
                ##############
                
                module.weight.data = attack_weight
                # print(attack_index)
                # self.attack_dict[name] = attack_weight
                # print("adding new entry in attack dict:")
                # print(self.attack_dict)

        # reset the bits2flip back to 0
        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0

        return attack_index, attack_name, attack_gradient
    

class oneshot_BFA(object):
    def __init__(self, criterion, k_top=10, flip_allowed = 1, layer_id = 0):

        self.criterion = criterion
        self.k_top = k_top
        self.n_bits2flip = k_top
        self.loss = 0
        self.flip_allowed = k_top
        self.layer_id = layer_id
        if flip_allowed <= k_top:
            self.flip_allowed = flip_allowed
    def flip_bit(self, name, m):
        '''
        the data type of input param is 32-bit floating, then return the data should
        be in the same data_type.
        '''
        if self.n_bits2flip == 0:
            return m.weight.data
        # 1. flatten the gradient tensor to perform topk (only count the grain wise biggest)
        if len(m.weight.grad.detach().size()) == 4:
            w_grad_topk, w_idx_topk = Get_grain_topk(m.weight.grad.detach().abs().permute(1, 2, 3, 0).contiguous().view(-1), self.k_top, self.flip_allowed, m.grain_size)
        else:
            w_grad_topk, w_idx_topk = Get_grain_topk(m.weight.grad.detach().abs().contiguous().view(-1), self.k_top, self.flip_allowed, m.grain_size)
        
        # update the b_grad to its signed representation
        if len(m.weight.grad.detach().size()) == 4:
            w_grad_topk = m.weight.grad.detach().permute(1, 2, 3, 0).contiguous().view(-1)[w_idx_topk]
            # print(w_grad_topk)
        else:
            w_grad_topk = m.weight.grad.detach().contiguous().view(-1)[w_idx_topk]

        # 2. create the b_grad matrix in shape of [N_bits, k_top]
        b_grad_topk = w_grad_topk * m.b_w.data

        # 3. generate the gradient mask to zero-out the bit-gradient
        # which can not be flipped
        b_grad_topk_sign = (b_grad_topk.sign() +
                            1) * 0.5  # zero -> negative, one -> positive
        # convert to twos complement into unsigned integer
        if len(m.weight.grad.detach().size()) == 4:
            w_bin = int2bin(m.weight.detach().permute(1, 2, 3, 0).contiguous().view(-1), m.N_bits).short()
        else:
            w_bin = int2bin(m.weight.detach().contiguous().view(-1), m.N_bits).short()
        w_bin_topk = w_bin[w_idx_topk]  # get the weights whose grads are topk
        # generate two's complement bit-map
        b_bin_topk = (w_bin_topk.repeat(m.N_bits,1) & m.b_w.abs().repeat(1,self.k_top).short()) \
        // m.b_w.abs().repeat(1,self.k_top).short()
        grad_mask = b_bin_topk ^ b_grad_topk_sign.short()

        # 4. apply the gradient mask upon ```b_grad_topk``` and in-place update it
        b_grad_topk *= grad_mask.float()

        # 5. identify the several maximum of absolute bit gradient and return the
        # index, the number of bits to flip is self.n_bits2flip
        grad_max = b_grad_topk.abs().max()
        _, b_grad_max_idx = b_grad_topk.abs().view(-1).topk(self.n_bits2flip)
        bit2flip = b_grad_topk.clone().view(-1).zero_()

        if grad_max.item() != 0:  # ensure the max grad is not zero
            bit2flip[b_grad_max_idx] = 1
            
            bit2flip = bit2flip.view(b_grad_topk.size())
        else:
            pass

        # 6. Based on the identified bit indexed by ```bit2flip```, generate another
        # mask, then perform the bitwise xor operation to realize the bit-flip.
        w_bin_topk_flipped = (bit2flip.short() * m.b_w.abs().short()).sum(0, dtype=torch.int16) \
                ^ w_bin_topk
        
        # 7. update the weight in the original weight tensor
        w_bin[w_idx_topk] = w_bin_topk_flipped  # in-place change
        if len(m.weight.grad.detach().size()) == 4:
            param_flipped = bin2int(w_bin, m.N_bits).contiguous().view(m.weight.data.size()[1], m.weight.data.size()[2], m.weight.data.size()[3], m.weight.data.size()[0]).permute(3, 0, 1, 2).float()
        else:
            param_flipped = bin2int(w_bin, m.N_bits).contiguous().view(m.weight.data.size()).float()

        return param_flipped
    def oneshot_attack_apply(self, model, data, target):
        ''' 
        Given the model, base on the current given data and target, go through
        all the layer and identify the bits to be flipped. 
        '''
        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
        model.eval()

        # 1. perform the inference w.r.t given data and target
        output = model(data)
        #         _, target = output.data.max(1)
        self.loss = self.criterion(output, target)
        
        
        # 2. zero out the grads first, then get the grads
        for name, m in model.named_modules():
            if ("conv" in name) or ("classifier" in name):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()
                    
        self.loss.backward()
        # 3. for each layer flip #bits = self.bits2flip
        counter = 0
        for name, module in model.named_modules():
            if ("conv" in name) or ("classifier" in name) or ("fc" in name) or ("downsample.0" in name):
                counter += 1
                # clean_weight = module.weight.data.detach()
                if counter == self.layer_id:
                    attack_weight = self.flip_bit(name, module)
                    module.weight.data = attack_weight
                elif self.layer_id == 0:
                    attack_weight = self.flip_bit(name, module)
                    module.weight.data = attack_weight
        # 4. if the loss_max does lead to the degradation compared to the self.loss,
        # then change the that layer's weight without putting back the clean weight
        # for name, module in model.named_modules():
            # attack_weight, attack_index = self.flip_bit(name, module)
            
            # attack_name = name
            
            # module.weight.data = attack_weight

        return
        
class RFA(object):
    def __init__(self, criterion, k_top=10, flip_allowed = 1, layer_id = 0):

        self.criterion = criterion
        self.n_bits2flip = k_top
        self.loss = 0
        self.layer_id = layer_id
    def flip_bit(self, name, m):
        '''
        the data type of input param is 32-bit floating, then return the data should
        be in the same data_type.
        '''
        if self.n_bits2flip == 0:
            return m.weight.data
        # 1. flatten the gradient tensor to perform topk (only count the grain wise biggest)
        if len(m.weight.detach().size()) == 4:
            w_grad_topk, w_idx_topk = Get_random_topk(m.weight.detach().abs().permute(1, 2, 3, 0).contiguous().view(-1), 1)
        else:
            w_grad_topk, w_idx_topk = Get_random_topk(m.weight.detach().abs().contiguous().view(-1), 1)
                # update the b_grad to its signed representation
        if len(m.weight.detach().size()) == 4:
            w_grad_topk = m.weight.detach().permute(1, 2, 3, 0).contiguous().view(-1)[w_idx_topk]
            # print(w_grad_topk)
        else:
            w_grad_topk = m.weight.detach().contiguous().view(-1)[w_idx_topk]

        # 2. create the b_grad matrix in shape of [N_bits, k_top]
        b_grad_topk = w_grad_topk * m.b_w.data

        # 3. generate the gradient mask to zero-out the bit-gradient
        # which can not be flipped
        b_grad_topk_sign = (b_grad_topk.sign() +
                            1) * 0.5  # zero -> negative, one -> positive
        # convert to twos complement into unsigned integer
        if len(m.weight.detach().size()) == 4:
            w_bin = int2bin(m.weight.detach().permute(1, 2, 3, 0).contiguous().view(-1), m.N_bits).short()
        else:
            w_bin = int2bin(m.weight.detach().contiguous().view(-1), m.N_bits).short()
        w_bin_topk = w_bin[w_idx_topk]  # get the weights whose grads are topk
        # generate two's complement bit-map
        b_bin_topk = (w_bin_topk.repeat(m.N_bits,1) & m.b_w.abs().repeat(1,1).short()) \
        // m.b_w.abs().repeat(1,1).short()
        # print(b_bin_topk.shape)
        grad_mask = b_bin_topk ^ b_grad_topk_sign.short()

        # 4. apply the gradient mask upon ```b_grad_topk``` and in-place update it
        b_grad_topk *= grad_mask.float()

        # 5. identify the several maximum of absolute bit gradient and return the
        # index, the number of bits to flip is 1
        grad_max = b_grad_topk.abs().max()
        _, b_grad_max_idx = b_grad_topk.abs().view(-1).topk(1)
        bit2flip = b_grad_topk.clone().view(-1).zero_()

        if grad_max.item() != 0:  # ensure the max grad is not zero
            bit2flip[b_grad_max_idx] = 1
            
            bit2flip = bit2flip.view(b_grad_topk.size())
            # print(b_grad_topk.size())
        else:
            bit2flip = bit2flip.view(b_grad_topk.size())
            pass

        # 6. Based on the identified bit indexed by ```bit2flip```, generate another
        # mask, then perform the bitwise xor operation to realize the bit-flip.
       
        w_bin_topk_flipped = (bit2flip.short() * m.b_w.abs().short()).sum(0, dtype=torch.int16) \
                ^ w_bin_topk
        
        # 7. update the weight in the original weight tensor
        w_bin[w_idx_topk] = w_bin_topk_flipped  # in-place change
        if len(m.weight.detach().size()) == 4:
            param_flipped = bin2int(w_bin, m.N_bits).contiguous().view(m.weight.data.size()[1], m.weight.data.size()[2], m.weight.data.size()[3], m.weight.data.size()[0]).permute(3, 0, 1, 2).float()
        else:
            param_flipped = bin2int(w_bin, m.N_bits).contiguous().view(m.weight.data.size()).float()

        return param_flipped
    def random_attack_apply(self, model):
        ''' 
        Given the model, base on the current given data and target, go through
        all the layer and identify the bits to be flipped. 
        '''
        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146

        # 1. for each layer flip #bits = self.bits2flip
        for t in range(self.n_bits2flip):
            counter = 0
            randint = random.randint(1, 20)
            for name, module in model.named_modules():
                if ("conv" in name) or ("classifier" in name) or ("fc" in name) or ("downsample.0" in name):
                    counter += 1
                    # clean_weight = module.weight.data.detach()
                    if counter == self.layer_id:
                        attack_weight = self.flip_bit(name, module)
                        module.weight.data = attack_weight
                    elif self.layer_id == 0:
                        if counter == randint:
                            attack_weight = self.flip_bit(name, module)
                            module.weight.data = attack_weight
                            

        return

class weight_shifter(object):
    def __init__(self, criterion, k_top=10, flip_allowed = 1, layer_id = 0):

        self.criterion = criterion
        self.n_bits2flip = k_top
        self.grain_size = [1,4]
        self.layer_id = layer_id
    def shuffle_weight(self, name, m):
        '''
        the data type of input param is 32-bit floating, then return the data should
        be in the same data_type.
        '''
        # 1. flatten the gradient tensor to perform topk (only count the grain wise biggest)
        device = torch.device('cuda:0')
        if len(m.weight.detach().size()) == 4:
            original_size = m.weight.detach().size()
            # print(original_size)
            reshaped_input = m.weight.detach().permute(1, 2, 3, 0).view(1, 1, -1, original_size[0])
            reshaped_orig = torch.zeros_like(reshaped_input)
            idx = np.random.choice(int(reshaped_input.size()[2]), 1, replace=False)
            idx2 = np.random.choice(int(reshaped_input.size()[3]/4), 1, replace=False)
            # print("Number of attack is %d" % reshaped_input.size()[3])
            for i in range(idx.shape[0]):
                for j in range(idx2.shape[0]):
                    reshaped_orig[0,0, idx[i] , 0 + 4*j] = reshaped_input[0,0, idx[i] , 0 + 4*j]
                    reshaped_input[0,0, idx[i] , 0 + 4*j] = reshaped_input[0,0, idx[i] , 3 + 4*j]
                    reshaped_input[0,0, idx[i] , 3 + 4*j] = reshaped_orig[0,0, idx[i] , 0 + 4*j]
                    reshaped_orig[0,0, idx[i] , 1 + 4*j] = reshaped_input[0,0, idx[i] , 1 + 4*j]
                    reshaped_input[0,0, idx[i] , 1 + 4*j] = reshaped_input[0,0, idx[i] , 2 + 4*j]
                    reshaped_input[0,0, idx[i] , 2 + 4*j] = reshaped_orig[0,0, idx[i] , 1 + 4*j]
            param_flipped = reshaped_input.view(original_size[1], original_size[2], original_size[3], original_size[0]).permute(3, 0, 1, 2)
            # print("Number of attack is %d" % reshaped_input.size()[3])
            # print(param_flipped[0,0,0,:4])
        else:
            original_size = m.weight.detach().size()
            reshaped_input = m.weight.detach().view(1, 1, original_size[0], original_size[1])
            reshaped_orig = torch.zeros_like(reshaped_input)
            idx = np.random.choice(int(reshaped_input.size()[2]), 1, replace=False)
            idx2 = np.random.choice(int(reshaped_input.size()[3]/4), 1, replace=False)
            # print("Number of attack is %d" % reshaped_input.size()[3])
            for i in range(idx.shape[0]):
                for j in range(idx2.shape[0]):
                    reshaped_orig[0,0, idx[i] , 0 + 4*j] = reshaped_input[0,0, idx[i] , 0 + 4*j]
                    reshaped_input[0,0, idx[i] , 0 + 4*j] = reshaped_input[0,0, idx[i] , 3 + 4*j]
                    reshaped_input[0,0, idx[i] , 3 + 4*j] = reshaped_orig[0,0, idx[i] , 0 + 4*j]
                    reshaped_orig[0,0, idx[i] , 1 + 4*j] = reshaped_input[0,0, idx[i] , 1 + 4*j]
                    reshaped_input[0,0, idx[i] , 1 + 4*j] = reshaped_input[0,0, idx[i] , 2 + 4*j]
                    reshaped_input[0,0, idx[i] , 2 + 4*j] = reshaped_orig[0,0, idx[i] , 1 + 4*j]
            param_flipped = reshaped_input.view(original_size[0], original_size[1])
            
        return param_flipped
    def shift_weight(self, name, m):
        '''
        the data type of input param is 32-bit floating, then return the data should
        be in the same data_type.
        '''
        # 1. flatten the gradient tensor to perform topk (only count the grain wise biggest)
        device = torch.device('cuda:0')
        if len(m.weight.detach().size()) == 4:
            original_size = m.weight.detach().size()
            # print(original_size)
            reshaped_input = m.weight.detach().permute(1, 2, 3, 0).view(1, 1, -1, original_size[0])
            pooling_result = torch.nn.functional.avg_pool2d(reshaped_input, self.grain_size, self.grain_size)
            
            random_permute = 68 * torch.randint(-1, 1, pooling_result.size())
            random_permute = random_permute.view(random_permute.size()[2:])
            # print(random_permute.size())
            random_permute = random_permute.unsqueeze(1).repeat(1,self.grain_size[0], 1).view(-1,random_permute.size()[1]).transpose(0,1)
            random_permute = random_permute.repeat(1, self.grain_size[1]).view(-1,random_permute.size()[1]).transpose(0,1)
            random_permute = random_permute.view(original_size[1], original_size[2], original_size[3], original_size[0]).permute(3, 0, 1, 2).float()
            random_permute = random_permute.to(device)
            # print(random_permute)
            param_flipped = m.weight.detach() + random_permute
        else:
            original_size = m.weight.detach().size()
            
            reshaped_input = m.weight.detach().view(1, 1, original_size[0], original_size[1])
            pooling_result = torch.nn.functional.avg_pool2d(reshaped_input, self.grain_size, self.grain_size)
            
            random_permute = 68 * torch.randint(-1, 1, pooling_result.size())
            random_permute = random_permute.view(random_permute.size()[2:])
            random_permute = random_permute.unsqueeze(1).repeat(1,self.grain_size[0], 1).view(-1,random_permute.size()[1]).transpose(0,1)
            random_permute = random_permute.repeat(1, self.grain_size[1]).view(-1,random_permute.size()[1]).transpose(0,1).float()
            random_permute = random_permute.to(device)
            param_flipped = m.weight.detach() + random_permute
        return param_flipped
    def random_attack_apply(self, model,option):
        ''' 
        Given the model, base on the current given data and target, go through
        all the layer and identify the bits to be flipped. 
        '''
        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
        counter = 0
        # 1. for each layer flip #bits = self.bits2flip
        for name, module in model.named_modules():
            if ("conv" in name) or ("classifier" in name) or ("fc" in name) or ("downsample.0" in name):
                counter += 1
                # clean_weight = module.weight.data.detach()
                if counter == self.layer_id:
                    if option == "weight_shift":
                        attack_weight = self.shift_weight(name, module)
                    elif option == "weight_shuffle":
                        attack_weight = self.shuffle_weight(name, module)
                    else:
                        assert False, 'No such weight manipulation option!'
                    module.weight.data = attack_weight
                elif self.layer_id == 0:
                    if option == "weight_shift":
                        attack_weight = self.shift_weight(name, module)
                    elif option == "weight_shuffle":
                        attack_weight = self.shuffle_weight(name, module)
                    else:
                        assert False, 'No such weight manipulation option!'
                    module.weight.data = attack_weight
        return