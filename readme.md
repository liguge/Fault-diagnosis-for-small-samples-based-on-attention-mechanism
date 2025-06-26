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
@article{He_2024, 
title={Interpretable modulated differentiable STFT and physics-informed balanced spectrum metric for freight train wheelset bearing cross-machine transfer fault diagnosis under speed fluctuations}, 
volume={62}, 
DOI={10.1016/j.aei.2024.102568}, 
journal={Advanced Engineering Informatics}, 
author={He, Chao and Shi, Hongmei and Li, Ruixin and Li, Jianbo and Yu, ZuJun}, 
year={2024}, 
pages={102568} 
}
```


# Our other works

```html
@article{HE,  
title = {Physics-informed interpretable wavelet weight initialization and balanced dynamic adaptive threshold for intelligent fault diagnosis of rolling bearings},  
journal = {Journal of Manufacturing Systems},  
volume = {70},  
pages = {579-592},  
year = {2023},  
issn = {1878-6642},  
doi = {https://doi.org/10.1016/j.jmsy.2023.08.014},  
author = {Chao He and Hongmei Shi and Jin Si and Jianbo Li}   
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
