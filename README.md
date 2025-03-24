# A Comprehensive Federated Learning Framework for Diabetic Retinopathy Grading and Lesion Segmentation

## Getting started

1. Create conda environment and install dependencies:

```shell
conda create -y -n TJDR-FL python=3.8

conda activate TJDR-FL

conda install -y pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=11.3 -c pytorch

pip install pip -U
pip install -r requirements.txt
```

2. Prepare dataset:

`dataset_dir="../TJDR-FL/task/datas/"`

- For `IDRiD`, `DDR-seg`, `DDR-cls`, and `APTOS2019` datasets, download the official dataset into the corresponding dir
  and subsequently run `../TJDR-FL/task/experiments/{dataset_name}.py`

- For `TJDR`, download the dataset from the link our provided and move all files into `../TJDR-FL/task/datas/TJDR`

## Run the code and Training

Code runs according to `*.yaml` config file, we provide two methods to run the code:

### 1. Run from base_config.yaml

Training form `configs/base_config.yaml` as follows:

```shell
cd task
python run.py  
```

**Note:** Please make sure `configs/base_config.yaml` is prepared as you expected, as the code will be executed based on
it in the case.

### 2. Run from base config templates

For convenience, we provide base config templates to perform training, in which code run
from `configs/base_configs/{classification|segmentation}/{dataset}.yaml`. You can copy the template to override the
`configs/base_config.yaml` to run the code.

Args:

```text
-b, --base_config_path : Specified the path of base config, default is base_config.yaml  , Optional
-g, --gpu              : Specified gpu to run, default is specified by config files      , Optional
-n, --network          : Enable Network to parallel computing, default is false
--all_gpu              : Enable all gpu to parallel computing, default is false
--host                 : Cloud host, default is initialized by 'configs/base_config.yaml', Optional
--port                 : Cloud port, default is initialized by 'configs/base_config.yaml', Optional
```

## Our TJDR dataset is available at [here](https://github.com/NekoPii/TJDR).

## System information

We train our model on 6 NVIDIA GeForce RTX 3090 GPUs with a 24GB memory per-card. Testing is conducted on the same
machines.

## Citation:
If you find this dataset is useful to your research, please cite our papers.
```text
@article{mao2024comprehensive,
  title={A Comprehensive Federated Learning Framework for Diabetic Retinopathy Grading and Lesion Segmentation},
  author={Mao, Jingxin and Ma, Xiaoyu and Bi, Yanlong and Zhang, Rongqing},
  journal={IEEE Transactions on Big Data},
  year={2024},
  publisher={IEEE}
}
```
