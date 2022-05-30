feature_dir=/data1/CoOpData/clip_feat/
# ImageNet OxfordPets OxfordFlowers StanfordCars Food101 Caltech101
for DATASET in ImageNet
do
    python linear_probe.py \
    --dataset ${DATASET} \
    --feature_dir ${feature_dir} \
    --num_step 8 \
    --num_run 3
done
