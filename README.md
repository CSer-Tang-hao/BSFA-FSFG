# BSFA-FSFG

## Boosting Few-shot Fine-grained Recognition with Background Suppression and Foreground Alignment (Under Review, 2022)
 
#### This code is only to give paper reviewers a verification and academic research. After the paper is accepted, we will polish and optimize the code.



## Requirements

 - `Python 3.6`
 - [`Pytorch`](http://pytorch.org/) >= 1.7.0 
 - `Torchvision` = 0.10
 - `scikit-image` = 0.18.1

## How to run

```bash

nohup python train.py --dataset [type of dataset] --model [backbone] --num_classes [num-classes] --nExemplars [num-shots]
nohup python test.py --dataset CUB-200-2011 --model R --num_classes 100 --nExemplars 5

# Example: run on CUB dataset, ResNet-12 backbone, 5 way 1 shot
nohup python train.py --dataset CUB-200-2011 --model R --num_classes 100 --nExemplars 1
nohup python test.py --dataset CUB-200-2011 --model R --num_classes 100 --nExemplars 1

```

[comment]: <> (### Data Preparation)

[comment]: <> (Download Datasets from:)

[comment]: <> (链接：https://pan.baidu.com/s/1Bevdjvf5xjroy3U-DA6w7Q )

[comment]: <> (提取码：ZZC3)

## Acknowledgement

This code is based on the implementations of [**fewshot-CAN**](https://github.com/blue-blue272/fewshot-CAN).

