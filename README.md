# StrucHIS

This repository contains source code and data for paper "Hierarchical Structure Sharing Empowers Multi-task Heterogeneous GNNs for Customer Expansion". (accepted by KDD 2025 Applied Data Science Track).


## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Running Experiments](#running-experiments)
- [Citation](#citation)
- [License](#license)


## Overview
<p align="center">
  <img src="images/StrucHIS-framework.pdf" height="200"/>
  <img src="images/StrucHIS-detail.pdf" height="200"/>
</p>
Figure 1: The overview framework of our StrucHIS. It is a heterogeneous graph-based Multi-task Learning framework that explicitly regulates structural knowledge sharing. It breaks down the structure learning phase into multiple stages and introduces sharing mechanisms at each stage, ensuring that task-specific requirements are addressed during each stage.


## Installation
Our model depends on PyTorch (CUDA toolkit if use GPU). You must have them installed before using our model
>
* Python 3.7
* Pytorch 1.10.2
* Cuda 11.1
* dgl > 0.7

You can adjust the versions of Python, PyTorch, and CUDA based on your setup, but make sure **the DGL version is above 0.7, otherwise some functions won't be available.** This command will install all the required libraries and their specified versions.
```python 
pip install -r requirements.txt
```

## Dataset
Our paper uses two public datasets and one private dataset from JD Logistics. Due to company security and privacy policies, we are unable to release the Logistics Dataset. Here, we provide two public datasets: **Aminer** and **DBLP** in the `Data` folder.

Note:The original DBLP data and preprocessing steps follow the method provided at https://github.com/THUDM/HGB/tree/master. However, in our version of DBLP, we removed the venue nodes and introduced **paper venue prediction** as a new task. As a result, the prediction performance on DBLP differs from what is reported in the HGB paper.



## Running Experiments
### Python command
For training and evaluating the model, run the following code
```python 
# Train on Aminer dataset
python python main.py --config our_Aminer
# Train on DBLP dataset
python  python main.py --config our_DBLP
```
Models are trained on the training set, halted when the model's performance on the validation set no longer improves after 40 epochs, and then evaluated on the test set. The train, validation, and test datasets have already been split in the `Data` directory.

The model will be run multiple times, and the final output will include the average value and the fluctuation range of the results.
  
Experiments were conducted on CentOS 7 with 1 24GB Tesla P40 GPUs

### Paramters
Basic parameters are set in `kg_lib\args.py`
>
* `config`: hyper-parameter file name
* `seed`: random seed

Hyper-parameters for each dataset are set in `kg_config\*.yaml`
>
* `rerun_num`: Number of repeated training runs for robustness. 
* `train_epoch`: Maximum number of training epochs.
* `early_stop`: Early stopping patience (in epochs).

>
* `learning_rate`: Learning rate used for optimization. 
* `weight_decay`   : Weight decay (L2 regularization).                
* `loss_weight`    : Weight assigned to each component of the loss function (e.g., [task1_weight, task2_weight]). 
* `save_metric`    : Metric used to save the best model (e.g., `'loss'`). 
* `eval_inter_ep`  : Frequency of evaluation during training (in epochs). 
>
* `layer_num`          : Number of heterogeneous graph layers in the model.              
* `node_feature_hid_len` : Hidden dimension for node feature representation. 
* `GAT_hid_len`        : Hidden dimension in heterogeneous graph layers. 
* `edge_feats_len`     : Dimension of edge feature representations.       
* `nhead`              : Number of attention heads.


### Best Model and Prediction Results
All results will be saved in the `Result` folder, including:

- `best_model_{i}.py`: The best model parameters saved from the *i*-th training run.
- `best_result.json`: Contains the test set performance of the best model from each training run.
- `log.txt`: Logs of the training process.



## Citation

If you use StrucHIS in your research, please cite the following paper:

> Xinyue Feng, Shuxin Zhong, Jinquan Hang, Wenjun Lyu, Yuequn Zhang, Guang Yang, Haotian Wang, Desheng Zhang, and Guang Wang. 2025. Hierarchical Structure Sharing Empowers Multi-task Heterogeneous GNNs for Customer Expansion. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD ’25), August 3–7, 2025, Toronto, ON, Canada. ACM, New York, NY, USA, 12 pages.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

