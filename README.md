# KIDA: Knowledge Inheritance in Dataset Aggregation
Zixuan Cao, Yang Xu, [Zhewei Huang](https://scholar.google.com/citations?user=zJEkaG8AAAAJ&hl=zh-CN&oi=sra), [Shuchang Zhou](https://scholar.google.com/citations?user=zYI0rysAAAAJ&hl=zh-CN&oi=sra)

This project releases our 1st place solution on [NeurIPS2021 ML4CO Dual Task](https://www.ecole.ai/2021/ml4co-competition/). The official competition report (including ours) is now on [arXiv](https://arxiv.org/abs/2203.02433). 

For this work, [arXiv](https://arxiv.org/abs/2201.10328), [Slide](https://github.com/megvii-research/NeurIPS2021-ML4CO-KIDA/blob/a5413e12b6e8d3d56b81096933198e94f692e517/ML4COPreV7.pdf) are available.

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

## Citation

```
@article{cao2022ml4co,
  title={ML4CO-KIDA: Knowledge Inheritance in Dataset Aggregation},
  author={Cao, Zixuan and Xu, Yang and Huang, Zhewei and Zhou, Shuchang},
  journal={arXiv preprint arXiv:2201.10328},
  year={2022}
}

@inproceedings{gasse2022machine,
  title={The Machine Learning for Combinatorial Optimization Competition (ML4CO): Results and Insights},
  author={Gasse, Maxime and Bowly, Simon and Cappart, Quentin and Charfreitag, Jonas and Charlin, Laurent and Ch{\'e}telat, Didier and Chmiela, Antonia and Dumouchelle, Justin and Gleixner, Ambros and Kazachkov, Aleksandr M and others},
  booktitle={NeurIPS 2021 Competitions and Demonstrations Track},
  pages={220--231},
  year={2022},
  organization={PMLR}
}
```

# Resouces
* [Technical Report](https://arxiv.org/abs/2201.10328)
* [Bilibili](https://www.bilibili.com/video/BV1nG411W71Y), [Youtube](https://www.youtube.com/watch?v=aFTkMKf77eU)
* [参赛方案介绍 - 旷视研究院](https://zhuanlan.zhihu.com/p/440726459)
