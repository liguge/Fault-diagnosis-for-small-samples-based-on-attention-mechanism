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
@article{ZHANG2022110242,  
title = {Fault diagnosis for small samples based on attention mechanism},  
journal = {Measurement},  
volume = {187},  
pages = {110242},  
year = {2022},  
issn = {0263-2241},  
doi = {https://doi.org/10.1016/j.measurement.2021.110242 },  
url = {https://www.sciencedirect.com/science/article/pii/S0263224121011507},  
author = {Xin Zhang and Chao He and Yanping Lu and Biao Chen and Le Zhu and Li Zhang}  
} 
```


# Our other works

```html
@ARTICLE{9374403,  
author={Luo, Hao and He, Chao and Zhou, Jianing and Zhang, Li},  
journal={IEEE Access},   
title={Rolling Bearing Sub-Health Recognition via Extreme Learning Machine Based on Deep Belief Network Optimized by Improved Fireworks},   
year={2021},  
volume={9},  
number={},  
pages={42013-42026},  
doi={10.1109/ACCESS.2021.3064962},  
url = {https://ieeexplore.ieee.org/document/9374403},  
}
```

# Environment

pytorch == 1.10.0  
python ==  3.8  
cuda ==  10.2   

# Contact
- **Chao He**
- **22110435#bjtu.edu.cn   (please replace # by @)**

## Views
![](http://profile-counter.glitch.me/liguge/count.svg)
