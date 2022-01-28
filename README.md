# KIDA: Knowledge Inheritance in Dataset Aggregation

This project releases our 1st place solution on [NeurIPS2021 ML4CO Dual Task](https://www.ecole.ai/2021/ml4co-competition/). 

[Arxiv](https://arxiv.org/abs/2201.10328), [Slide](https://drive.google.com/file/d/1O8T1Jv6CE_fQDnYdcZ6ftFEZPXabOVay/view?usp=sharing) and [model weights](https://drive.google.com/drive/folders/1WuLt7ww-c45tdQ1DnE3vyozq3i1UORg9?usp=sharing) are available.

#  Environment Setup
Follow the [tutorials](https://github.com/ds4dm/ml4co-competition/blob/main/START.md) of [ml4co-competition](https://github.com/ds4dm/ml4co-competition) to download ML4CO instance dataset and set up your Python dependencies.  Suppose the directory of instance dataset is `/YOUR_PATH/ml4co-competition/instances`. The dependencies of KIDA can be found in  `./Nuri/conda.yaml` . Anaconda environment of ml4co can be created by `source init.sh`

# Testing
Download [model weights](https://drive.google.com/drive/folders/1WuLt7ww-c45tdQ1DnE3vyozq3i1UORg9?usp=sharing) and put them in `/YOUR_PATH/NeurIPS2021-ML4CO-KIDA/Nuri/` .

Copy the submission files to ml4co-competition directory.

```
cp -r /YOUR_PATH/NeurIPS2021-ML4CO-KIDA/Nuri/ /YOUR_PATH/ml4co-competition/submissions/
```
Evaluate models in dual task by running the following commands.


```bash
cd /YOUR_PATH/ml4co-competition/submissions/Nuri
conda activate ml4co
python ../../common/evaluate.py dual item_placement
python ../../common/evaluate.py dual load_balancing
python ../../common/evaluate.py dual anonymous
```


# Training
Modify the configuration file in `./train_files/configs` to meet the needs of the local environment.  

`./train_files/configs/dataset.ini` is the configuration file for data generation process. `DATASET_DIR` decides  the directory where the instance dataset is stored. `STORE_DIR`  decides the directory where generated data is stored. `NODE_RECORD_PROB` decides the probability of using Strong Branching when collecting data. `TIME_LIMIT` decides the time limit for SCIP solver. `POLICY_TYPE` decides the  model we use (0 for Item Placement Benchmark, 1 for Workload Apportionment Benchmark and 2 for Anonymous Benchmark).

`./train_files/configs/train.ini` is the configuration file for training process. `STORE_DIR` decides the directory where training files are stored (It must keep the same as `STORE_DIR` in `dataset.ini`). `TRAIN_NUM` and `VALID_NUM` decide the number of data used in training process for each epoch. 

Then, run the following commands to train the model.


```bash
cd /YOUR_PATH/NeurIPS2021-ML4CO-KIDA/train_files
# Train Item Placement Benchmark
source item.sh
# Train Workload Apportionment Benchmark 
source load.sh
# Train Anonymous Benchmark
source ano.sh
```

# Resouces
[参赛方案介绍 - 旷视研究院](https://zhuanlan.zhihu.com/p/440726459)
