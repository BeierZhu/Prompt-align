# sh feat_extractor.sh
DATA=/data1/CoOpData
OUTPUT='/data1/CoOpData/clip_feat/'
SEED=1

GPULIST=(0 1 2 3)
GPUIDX=0

# oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet
# imagenet oxford_pets oxford_flowers stanford_cars food101 caltech101
for DATASET in imagenetv2 imagenet_sketch imagenet_a imagenet_r
do
    for SPLIT in train val test
    do
        while true 
        do 
            sleep 10
            let STATIDX=GPULIST[GPUIDX]+2
            stat=$(gpustat | awk '{print $11}' | sed -n ${STATIDX}'p')
            if [ "$stat" -lt 20 ]
            then
                break
            fi 
            let GPUIDX=(GPUIDX+1)%${#GPULIST[@]}
            echo $GPUIDX'N'
        done
        CUDA_VISIBLE_DEVICES=${GPULIST[${GPUIDX}]} python feat_extractor.py \
        --split ${SPLIT} \
        --root ${DATA} \
        --seed ${SEED} \
        --dataset-config-file ../configs/datasets/${DATASET}.yaml \
        --config-file ../configs/trainers/CoOp/rn50_val.yaml \
        --output-dir ${OUTPUT} \
        --eval-only &
        sleep 10
    done
done
