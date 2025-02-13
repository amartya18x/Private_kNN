# Overview


This repository provides the implementation for the paper Private-kNN : Practical Differential Privacy for Computer Vision by Yuqing Zhu, Xiang Yu, Manmohan Chandraker, Yu-Xiang Wang.

The Private-kNN is a data-efficient algorithm for differentially private (DP) deep learning under the ``knowledge transfer `` framework. It represents the first practical solution that addresses this important problem scales to larger models while preserving theoretically meaning full DP guarantee (\epsilon < 1).


It reduces more than 90% of the privacy loss compared to the state-of-the-art PATE method on SVHN tasks.

Dataset                 | Methods | Queries | \epsilon| Accuracy | NP Accuracy
---------------------- | :------------------------: | :--------:| :---:   | :---: |:---:   |
MNIST      | LNMAX     | 1000     |  8.03 |  98.1% | 99.2%  | 
MNIST      | GNMAX     | 286      |  1.97 | 98.5%  |  99.2% |
MNIST      | **Private-kNN**| 735 |  **0.47** |   98.8%|   99.2%| 
|||||||
SVHN      | LNMAX | 100 |  8.19 |  90.1% | 92.8%  | 
SVHN  | GNMAX     | 3098    |  4.96 | 91.6%  |  92.8% |
SVHN      | **Private-kNN**| 2939|  **0.49** |   91.6%|   92.8%|   


Evaluations on two realistic identity relevant tasks --- face attribute classification on **Celeb-A** and human attribute classification on **Market 1501**.

Dataset                 | Methods | Queries | \epsilon| Accuracy | NP Accuracy
---------------------- | :------------------------: | :--------:| :---:   | :---: |:---:   |
CelebA     | GNMAX            | 600     |  7.72 |  85.0%| 89.5%  | 
CelebA     | **Private-kNN**  |  800    |  1.24 | 84.70%* |  89.5% |
Market1501 |GNMAX             | 800 |  13.41 |   86.8%|   92.1%|  
Market1501 | **Private-kNN**  | 1200 |  **1.38** |   89.2%|   92.1%|  



## Dependencies

This model uses Pytorch as deep learning framework and apply autodp for privacy analysis. To install autodp with the correct version,

```git clone  https://github.com/yuxiangw/autodp``` and put it in the `Private_kNN` folder.
```cd autodp,  git checkout 75f33d9``` 


## How to Play ?

We train the private-kNN with an iterative process and finally release a student model in the public domain. Each iteration consists of two steps: 1) Update the feature extractor of a kNN model. Train a student model in the public domain. 

1) Update the feature extractor for private-kNN: We initialize the feature extractor with a public extractor --- Histogram of Oriented Gradient (HOG) features. In the next iteration, we use the neural network of the last iteration student model (except for the last softmax layer) to update the feature extractor. Note that this interactive scheme will iteratively refine the feature embedding
used by kNN without using any private information.

2) Train a student model: Once the feature extractor of Private-kNN is updated, we train a student model by labeling a limited number of student queries (the public data) with pseudo-labels. For each student query, wefirst generate a random subset from the entire private domain, and then pick the k 
 nearest neighbors among the subset. The pseudo-label is generated with private voting of k neighbors, and the detailed private aggregation process can be found in the main paper. 

We prepare the code of MNIST and SVHN task in **digit_pytorch** folder which contains several Python files that you will need to edit

`svhn_config.py `| Defines the configure of the code, which needs to edit manually in each iteration. 
 
 
In the first iteration, we set config.extract_feature = 'hog' to initialize the feature extractor and change to 
 'feature' model in the later iterations. 

`knn.py` | The main procedure of the private-kNN algorithm. To train a student model, use the following command:
`
python knn.py
`

`aggregation.py` | Defines the  noisy screening and noisy aggregation processes which are two core components of the privacy guarantee.

Instructions:
1) set `config.extract_feature = hog`, run `python knn.py` | Train a student model based on HOG features
2) (Optional） Apply semi-supervised training (UDA or VAT) to train a better student model
3) set `config.extract_feature = feature`, run `python knn.py` | Update the feature extractor with the student model in the last iteration.

Repeat 2, 3 steps, usually the model converges in two iterations.
## Semi-supervised training with the student model

In the paper, we use UDA to train the student model for SVHN and CIFAR-10 tasks, which allows us 
to save the privacy budget with a limited number of student queries.

Leveraging UDA to label the entire public domain (except for the testing) requires us first clone the [UDA repository](https://github.com/google-research/uda), which looks like `clean_kNN/uda`.
Then we set `config.use_uda = True` in  `clean_kNN/pate_pytorch/svhn_config.py`, which allows us to save answered queries (data & pseudo-labels) and the remaining unlabeled data into
the `uda/log` folder. Then you could conduct semi-supervised training and save the pseudo-labels of the entire public set into `uda.py`.
 
Further, we could train a student model and update the feature extractor by specify `config.uda_path = uda_path, config.use_uda_data = True` and run 
`python knn.py` again.




## Details with CelebA datasets
 We use ImageNet pretrained ResNet50m to extract feature for private-kNN. The default setting is: k=800, sigma=100, gamma (sampling ratio)=0.05 and 
the number of queries to answer is 800. We report the privacy and accuracy after one iteration (based on ResNet feature only). Moreover iterations can be done to further improve the accuracy.

## Privacy analysis


We use [autodp](https://github.com/yuxiangw/autodp) --- an automating differential privacy computation for the privacy analysis. The privacy analysis consists of the noisy screening and the noisy aggregation.
In the configuration file, `config.sigma1` and `config.threshold` specify the noisy scale and the threshold for the noisy screen. `config.gau_scale` denotes the noisy scale of the noisy aggregation.  

Autodp supports a RDP (Renyi Differential Privacy) based analytical Moment Accountant, which allows us to track the RDP for each query conveniently.
We first declare the moment accountants in `knn.py`:
```angular2html
    acct = rdp_acct.anaRDPacct()
```
We next define the CGF functions of both the noisy screening and the noisy aggregation.

```angular2html
    gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': config.sigma}, x)
    gaussian2 = lambda x: rdp_bank.RDP_inde_pate_gaussian({'sigma': config.gau_scale}, x)

```
Then we track RDP of the noisy screening over |teachers_preds| queries using the following command. 
```angular2html
    acct.compose_poisson_subsampled_mechanisms(gaussian, prob,coeff = len(teachers_preds))
    acct.compose_poisson_subsampled_mechanisms(gaussian2, prob,coeff = len(stdnt_labels))
```
`prob` is the sampling ratio and `len(stdnt_labels)` is the number of answered queries. We only count the privacy loss of the noisy aggregation on those answered queries.

Finally, we compute the privacy loss with a given `config.delta` with
```angular2html
    print(acct.get_eps(config.delta))
```
This paper proposes a new RDP analysis of the noisy screening process, a detailed comparison between several noisy screening methods can be found in the `private_kNN/privacy_analysis` folder.
