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

# reg_factor=0.001
reg_factor=0.0
ingrain_d2=1
inbit=1
inM2D=0.0
resgrain_d2=1
resbit=1
resM2D=0.0
outgrain_d2=1
outbit=1
outM2D=0.0

label_info=i${ingrain_d2}r${resgrain_d2}o${outgrain_d2}_adam${resM2D}_${resbit}bit

$PYTHON main.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${label_info} \
    --input_grain_size 1 ${ingrain_d2} --input_num_bits ${inbit} --input_M2D ${inM2D} \
    --res_grain_size 1 ${resgrain_d2} --res_num_bits ${resbit} --res_M2D ${resM2D} \
    --output_grain_size 1 ${outgrain_d2} --output_num_bits ${outbit} --output_M2D ${outM2D} \
    --epochs ${epochs} --learning_rate 0.001 \
    --optimizer ${optimizer} \
    --schedule 80 120 160  --gammas 0.1 0.1 0.5 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id ${gpuid} \
    --print_freq 100 --decay 0.000001 \
    --regular_factor ${reg_factor}
    
$PYTHON test_main.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --resume ./save/${DATE}/${dataset}_${model}_${epochs}_${label_info}/model_best.pth.tar \
    --input_grain_size 1 ${ingrain_d2} --input_num_bits ${inbit} --input_M2D ${inM2D} \
    --res_grain_size 1 ${resgrain_d2} --res_num_bits ${resbit} --res_M2D ${resM2D} \
    --output_grain_size 1 ${outgrain_d2} --output_num_bits ${outbit} --output_M2D ${outM2D} \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id ${gpuid}
    