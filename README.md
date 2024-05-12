

<div align="center">

<h2><a href="">Kolmogorov-Arnold Network (KAN) for Recommendations </a></h2>


</div>
<p align = "justify"> 
Official Implementation of Kolmogorov-Arnold Network (KAN) for Recommendations. Any communications, collaborations, issues, PRs are welcomed. The contributors will be listed [here](https://github.com/TianyuanYang/KAN4Rec?tab=readme-ov-file#Contributors). Please contact yueliu19990731@163.com or tianyuan.yang@u.nus.edu. If you find this repository useful to your research or work, it is really appreciate to star this repository. :heart:
</p>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Usage">Usage</a></li>
    <li><a href="#acknowledgement">Acknowledgement</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>
</details>


## Usage

### Datasets

MovieLens-1m and MovieLens-20m.




| datasets         | #users            |   #items   |    #actions    | average length | density |
| --------------- | --------------- | :---------: | :-----------: | :------------------: | :-------: |
| ML-1m            | 6040  |    3416    |    1m    |        163.5        |    4.79%     |
| ML-20m        |   138,493 |    26,744   |     20m    |       144.4         |     0.54%     |




### Requirements

codes are tested on Python3.8.16 and 1 NVIDIA Tesla V100 SXM2 16 GB

```
numpy==1.23.5
pandas==1.5.3
scipy==1.9.1
torch==2.0.0
tqdm==4.65.0
wget==3.2
```

### Quick Start
```
python main.py --template train_kan4rec --lr 1e-2 --dataset_code ml-1m
```

### Results
#### ML-1m
| **NDCG** | BERT4Rec | KAN4Rec |
| --- | --- | --- |
| **@1** | 0.3445  | **0.3499** | 
| **@5** | 0.5068 | **0.5133** |
| **@10** | 0.5417 | **0.5477** |
| **@20** | 0.5657 | **0.5719** |
| **@50** | 0.5875 | **0.5932** |
| **@100** | 0.5937 | **0.5991** |

|**Recall** | BERT4Rec | KAN4Rec |
|--- | --- | --- |
|**@1** | 0.3445 | **0.3499** |
| **@5** | 0.6517 | **0.6560** |
| **@10** | 0.7590 | **0.7622** |
| **@20** | 0.8535 | **0.8575** |
| **@50** | 0.9622 | **0.9635** |
| **@100** | 0.9997 | 0.9997 |

## Acknowledgements

Our code are partly based on the following GitHub repository. Thanks for their awesome works. 
- [BERT4Rec-VAE-Pytorch](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch): the implement of BERT4Rec model (PyTorch version).
- [fast-kan](https://github.com/ZiyaoLi/fast-kan): the implement of KAN (fast version). 


## Citations

If you find this repository helpful, please cite our paper (coming soon).

# Contributors

<a href="https://github.com/TianyuanYang" target="_blank"><img src="https://avatars.githubusercontent.com/u/53520309?v=4" alt="TianyuanYang" width="96" height="96"/></a> <a href="https://github.com/yueliu1999" target="_blank"><img src="https://avatars.githubusercontent.com/u/41297969?s=64&v=4" alt="yueliu1999" width="96" height="96"/></a> 


<p align="right">(<a href="#top">back to top</a>)</p>
