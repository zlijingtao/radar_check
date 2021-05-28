#!/usr/bin/env sh
PYTHON="/opt/anaconda3/envs/pytorch_p36/bin/python"
############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
    mkdir ./save/${DATE}/
fi

############ Configurations ###############
model=quan_resnet20
dataset=cifar10
epochs=200
batch_size=128
optimizer=Adam
gpuid=0

ingrain_d2=1
inbit=1
inM2D=0.0
resgrain_d2=1
resbit=1
resM2D=0.0
outgrain_d2=1
outbit=1
outM2D=0.0

resume_path="./quan_resnet20_cifar10.pth.tar"

    
test_batch_size=256
attack_sample_size=128 # number of data used for PBFA
n_iter=2 # number of iteration to perform PBFA (actuall total number of bits flipped in PBFA)
k_top=40 # Number of bits that is taken into candidates (PBFA)/ flipped per layer(RBFA/Oneshot): PBFA set to 40, RBFA set to 20, Oneshot set to 1.
layer_id=0 #{1~20}indicate which layer to attack, set to 0 indicates attacking every layer with the same rate

massive=10
#clean up below, check_factor
check_gsize=8
check_bit=2
limit_row=10
$PYTHON test_main.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --resume ${resume_path} --check --massive ${massive} --check_gsize ${check_gsize} --check_bit ${check_bit} --limit_row ${limit_row}\
    --input_grain_size 1 ${ingrain_d2} --input_num_bits ${inbit} --input_M2D ${inM2D} \
    --res_grain_size 1 ${resgrain_d2} --res_num_bits ${resbit} --res_M2D ${resM2D} \
    --output_grain_size 1 ${outgrain_d2} --output_num_bits ${outbit} --output_M2D ${outM2D} \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id ${gpuid} \
    --layer_id ${layer_id} \
    --reset_weight \
    --n_iter ${n_iter} --k_top ${k_top} \
    --attack_sample_size ${attack_sample_size} --bfa
    
        