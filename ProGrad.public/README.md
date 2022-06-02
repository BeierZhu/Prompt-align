# How to Run

## GPU memory needed

All the experiments is able to run on a single graphic card. However, **if you want to get results on ImageNet, the memory on any single graphic card should be larger than 24 GB.** Around 12 GB is enough for other datasets. 


## How to Install
This code is built on top of the toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). But we have some modification on it. So please install the provided Dassl.ProGrad.pytorch. Go the the folder Dassl.ProGrad.pytorch provided in the appendix, and prepare the environment as follows:

```
# Create a conda environment
conda create -n dassl python=3.7

# Activate the environment
conda activate dassl

# Install dependencies
pip install -r requirements.txt

# Install torch (version >= 1.7.1) and torchvision
# Please make sure you have installed the gpu version due to the speed.
# For example:
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

After that, run `pip install -r requirements.txt` under `ProGrad.public/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.

## Few-shot setting on 11 datasets

Basic format:
```
bash main.sh ${DATASET_NAME} ${CONFIG_NAME} end ${CONTEXT_TOKENS_NUMBER} ${SHOTS} False
```

For example, to run 1, 2, 4, 8, and 16 shots on stanford_cars, 
**CLIP + CoOp (M=16, end)**:

- 1 shot: `bash main.sh stanford_cars rn50_ep50 end 16 1 False`
- 2 shots: `bash main.sh stanford_cars rn50_ep100 end 16 2 False`
- 4 shots: `bash main.sh stanford_cars rn50_ep100 end 16 4 False`
- 8 shots: `bash main.sh stanford_cars rn50 end 16 8 False`
- 16 shots: `bash main.sh stanford_cars rn50 end 16 8 False`

**CLIP + CoOp + ProGrad**:

**Please take note that the 8-shots and 16-shots results on Flowers102, DTD, and EuroSAT are gotten with lambda as 0.8.** To get the results in our paper, please change the variable LAMBDA in prograd.sh from 1.0 to 0.8.

- 1 shot: `bash prograd.sh stanford_cars rn50_ep50 end 16 1 False`
- 2 shots: `bash prograd.sh stanford_cars rn50_ep100 end 16 2 False`
- 4 shots: `bash prograd.sh stanford_cars rn50_ep100 end 16 4 False`
- 8 shots: `bash prograd.sh stanford_cars rn50 end 16 8 False`
- 16 shots: `bash prograd.sh stanford_cars rn50 end 16 16 False`


```
output
|–– caltech101/
|   |–– CoOp/
|   |   |–– rn50_16shots/
|   |   |   |–– nctx16_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
|   |   |–– rn50_8shots/
|   |   |   |–– nctx16_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
```

To calculate the average results for the folder `rn50_16shots/nctx16_cscFalse_ctpend/`, you can run

```bash
python parse_test_res.py output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend
```

Then, you will see something like this in your terminal

```bash
Parsing files in output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend
file: output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend/seed1/log.txt. accuracy: 91.81%. error: 8.19%.
file: output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend/seed2/log.txt. accuracy: 92.01%. error: 7.99%.
file: output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend/seed3/log.txt. accuracy: 92.17%. error: 7.83%.
===
Summary of directory: output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend
* accuracy: 92.00% +- 0.15%
* error: 8.00% +- 0.15%
===
```

**How to visualize nearest words for the learned context tokens?** All you need is `interpret_prompt.py`. Say the learned tokens are saved in `a/b/c/prompt_learner/model.pth.tar` and you would like to see the top-3 nearest words for each token. In this case, run `python interpret_prompt.py a/b/c/prompt_learner/model.pth.tar 3`

## Robustness to Distribution Shift
To reproduce the robustness experiments, you can simply load the models learned on ImageNet and evaluate them on the following datasets: `imagenetv2`, `imagenet-sketch`, `imagenet-a` and `imagenet-r`.

The command is provided in `scripts/eval.sh`. The key arguments are `--model-dir`, `--load-epoch` and `--eval-only`. `--model-dir` indicates the directory where the models are saved (i.e. the entire folder containing `log.txt`, the tensorboard file and `prompt_learner/`). `--load-epoch` tells the code to load the model saved at a specific epoch, like `--load-epoch 50` for ImageNet for more details).

For example, to evaluate `CLIP + CoOp (M=16, end)` on ImageNetV2, you can do

```bash
# Don't need to use rn5_ep50 here as no training is performed
bash eval.sh imagenetv2 rn50
```

If you want to get the results of our method, simply change the TRAINER to `ProGrad`.

The default setting is `SHOTS=4`. Feel free to modify the script.

Again, you can use `parse_test_res.py` to automate the calculation of average performance. This time you should append `--test-log`, e.g., `python parse_test_res.py directory --test-log`.

## Zero-Shot CLIP
See `CoOp/scripts/zeroshot.sh`.

## Generalization From Base to New Classes

You will need `base2new_train_main.sh`, `base2new_test_main.sh`, `base2new_train_prograd.sh`, and `base2new_test_prograd.sh`. The scripts with the prefix `base2new_train` train a model on base classes while the ones with the prefix `base2new_test` evaluate the trained model on new classes. Both kinds of scripts have only one input argument, i.e., `DATASET`. `DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `CoOp/configs/datasets/`.

The scripts with postfix `prograd.sh` are used for our proposed method, while the ones with the postfix `main.sh` are used for CoOp.

Below we provide an example on how to evaluate the model on ImageNet.

```bash
bash base2new_train_prograd.sh stanford_cars
bash base2new_test_prograd.sh stanford_cars
```
**If you want to test results on ImageNet, remember to change the CFG from "rn50_ep100" to "rn50_ep50", and change the LOADEP from 100 to 50 in the corresponding script.**

When the evaluation is done, you can use `parse_test_res.py` to automatically calculate the average results. For instance, after you finish the evaluation using the aforementioned commands, you would get

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– stanford_cars/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– rn50_ep100/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– stanford_cars/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– rn50_ep100/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Then, to get the average performance on the base classes, run

```bash
python parse_test_res.py output/base2new/train_base/stanford_cars/shots_16/CoCoOp/rn50_ep100
```

To get the average performance on the new classes, run

```bash
python parse_test_res.py output/base2new/test_new/stanford_cars/shots_16/CoCoOp/rn50_ep100 --test-log
```

