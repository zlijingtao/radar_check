import torch
import numpy as np
from itertools import combinations

def place_ones(size, count):
    for positions in combinations(range(size), count):
        p = [-1] * size

        for i in positions:
            p[i] = 1

        yield p


def get_code_book(grain_size, limit_row):
    '''grain size is a 2 element tuple'''
    if grain_size[1] > 16:
        print("code_book of size larger than 16 is not supported, start using 16 codebook to match check_gsize of {}.".format(grain_size[1]))
        comb = list(place_ones(16, 8))
        assert grain_size[1]%16 == 0 , 'Fail to generate codebook, check_gsize must be integer multiples of 16!'
        factor = int(grain_size[1]/16)
        if (limit_row > 0) and limit_row*factor < len(comb):
            comb_base = comb[:limit_row]
            for i in range(factor-1):
                comb_in = comb[limit_row*(i+1):limit_row*(i+2)]
                comb_base = [x + y for x, y in zip(comb_base, comb_in)]
            comb = comb_base
                
        else:
            assert False, "limit_row of size {} is not supported by current setting".format(limit_row)
        # np.concatenate([a, a], axis=1).tolist()
    else:
        comb = list(place_ones(grain_size[1], int(grain_size[1]/2.0)))
        if (limit_row > 0) and limit_row < len(comb):
            comb = comb[:limit_row]
            
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # comb = torch.FloatTensor(comb).short().to(device)
    comb = torch.FloatTensor(comb).char()
    return comb

# comb = get_code_book([1,8], 0)
# print(comb)

# print(list(place_ones(8, 4)))