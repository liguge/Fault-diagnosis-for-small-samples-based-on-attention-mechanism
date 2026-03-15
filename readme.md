# DCA-BiGRU
The pytorch implementation of the paper [Fault diagnosis for small samples based on attention mechanism](https://doi.org/10.1016/j.measurement.2021.110242)

**However, in fact, the title [Fault diagnosis for small samples based on interpretable improved space-channel attention mechanism and improved regularization algorithms](https://doi.org/10.1016/j.measurement.2021.110242) fits the research content of the paper better.**

The dataset comes from 12khz, 1hp

![微信图片_20211204105938](https://user-images.githubusercontent.com/19371493/144694599-2e79190d-40cb-455e-95cf-a1da552cb707.png)

# Contributions:
1. **1D-signal attention mechanism** [[code](https://github.com/liguge/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/blob/main/oneD_CS_attention.py)]
2. **AMSGradP**   [[code](https://github.com/liguge/AMSGradP-for-intelligent-fault-diagnosis)]
3. **1D-Meta-ACON** [[code](https://github.com/liguge/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/blob/main/oneD_Meta_ACON.py)]
4. **At the beginning, I found that many model designs did not connect GAP operation after BiGRU/BiLSTM, which is the basically routine operation. I found that GAP works very well.**  [[code](https://github.com/liguge/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/blob/beb35522b283853aa12390721136583bb09821bf/model_train.py#L119)]
5. **1D-Grad-CAM++** [[code](https://github.com/liguge/1D-Grad-CAM-for-interpretable-intelligent-fault-diagnosis)]
6. **AdaBN** [[code](https://github.com/liguge/Fault-diagnosis-for-small-samples-based-on-attention-mechanism/blob/main/adabn.py)]
# Attention Block（SCA）
![1-s2 0-S0263224121011507-gr5_lrg](https://user-images.githubusercontent.com/19371493/160417827-560103d1-0ebc-4340-bcba-c5977ba78bf7.jpg)

# How does it work?

![微信图片_20220422112054](https://user-images.githubusercontent.com/19371493/164590358-4a2b1c84-20ee-4477-a217-0a2487170831.png)


# If it is helpful for your research, please kindly cite this work:


```html
@article{he2025pifcapsule,
  title={Human prior knowledge-embedded first-layer interpretable paradigm for rail transit vehicle human-computer collaboration monitoring},
  author={He, Chao and Shi, Hongmei and Liao, Jing-Xiao and Liu, Qiuhai and Li, Jianbo and Yu, Zujun},
  journal={Journal of Industrial Information Integration},
  volume={51},
  pages={101068},
  year={2025},
  doi={10.1016/j.jii.2026.101068},
  publisher={Elsevier}
}
```


# Our other works

```html
@article{HE2023110846,
title = {IDSN: A one-stage interpretable and differentiable STFT domain adaptation network for traction motor of high-speed trains cross-machine diagnosis},
journal = {Mechanical Systems and Signal Processing},
volume = {205},
pages = {110846},
year = {2023},
doi = {https://doi.org/10.1016/j.ymssp.2023.110846},
author = {Chao He and Hongmei Shi and Jianbo Li},
} 
```

# Environment

pytorch == 1.10.0  
python ==  3.8  
cuda ==  10.2   

# Contact
- **Chao He**
- **chaohe#bjtu.edu.cn   (please replace # by @)**

## Views
![](http://profile-counter.glitch.me/liguge/count.svg)
