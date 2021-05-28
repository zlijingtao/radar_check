#### Models for CIFAR-10 ############

from .quan_resnet_cifar import quan_resnet20, quan_resnet32, quan_resnet44, quan_resnet56

from .quan_resnet_cifar_check import quan_resnet20_c, quan_resnet32_c, quan_resnet44_c, quan_resnet56_c

from .vanilla_resnet_cifar import resnet20, resnet32, resnet44, resnet56

from .ResNet_quan import resnet18b_quan, resnet34b_quan, resnet50b_quan, resnet101b_quan
from .ResNet_quan_check import resnet18b_quan_c, resnet34b_quan_c, resnet50b_quan_c, resnet101b_quan_c

from .resnet_vanilla import resnet18, resnet34, resnet50, resnet101, resnet152
