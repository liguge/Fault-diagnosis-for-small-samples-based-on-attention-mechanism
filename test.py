import numpy as np
import torch
import os
import re
import scipy.io as scio
import scipy.signal
from torch.utils import data as da
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#########数据载入模块################
raw_num = 40
####################向数据中添加噪声不同信噪比的噪声#######################
class Data(object):
    '''
    读取mat格式数据，由于每个故障数据数量不同，这里只截取前480000个数据
    get_data()产生的数据为（2400，2000）的输入数据
    get_label()产生的数据为（2400，1）的标签数据
    '''

    def __init__(self):
        self.data = self.get_data()
        # 获取数据
        self.label = self.get_label()
        # 获取数据标签

    def file_list(self):
        return os.listdir('./data/')

    # 返回该目录下的所有文件夹
    def get_data(self):
        file_list = self.file_list()
        x = np.zeros((1024, 0))
        # print(file_list)
        for i in range(len(file_list)):
            file = scio.loadmat('./data/{}'.format(file_list[i]))
            for k in file.keys():
                file_matched = re.match('X\d{3}_DE_time', k)
                # file_matched = re.match('unnamed', k)
                if file_matched:
                    key = file_matched.group()
            # if i == 0:
            data1 = np.array(file[key][80000:96624])
            for j in range(0, len(data1) - 1023, 400):  # sliding window: 30000, step: 3000
                x = np.concatenate((x, data1[j:j + 1024]), axis=1)
            # else:
            # data = np.hstack((data, file[key][0:10]))
        return x.T

    def get_label(self):
        file_list = self.file_list()
        title = np.array([i.replace('.mat', '') for i in file_list])
        # 字典解析式   file_list是一个字典，遍历这个字典。将i中包含.mat的字段用‘’来替换,replace在python中只能用来替换字符串
        # 用空格来替换掉.mat这个文件格式
        label = title[:, np.newaxis]
        # 将一维数组转为一维矩阵，方便处理
        label_copy = np.copy(label)
        for _ in range(raw_num - 1):
            label = np.hstack((label, label_copy))
            # 生成标签数据，横向堆叠很多次，一定要记得转换成数组
        return label.flatten()


Data = Data()
data = Data.data
data_f = Data.data
label = Data.label
y = label.astype("int32")
# # lb = LabelBinarizer()
# # y = lb.fit_transform(label)   #标签二值化
# # dataNew = "G:\\研究生资料\\研二\\实践code(pytorch)\\datanew.mat"
# # scio.savemat(dataNew,mdict={'data':data,'label':label})
# ##############################数据归一化处理###########################
ss = MinMaxScaler()
data = data.T
data = ss.fit_transform(data).T
############################划分训练集和测试集#######################
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state=2)
X_train = torch.from_numpy(X_train).unsqueeze(1)
X_test = torch.from_numpy(X_test).unsqueeze(1)


class TestDataset(da.Dataset):
    def __init__(self):
        self.Data = X_train
        self.Label = y_train

    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label

    def __len__(self):
        return len(self.Data)


Test = TestDataset()
test_loader = da.DataLoader(Test, batch_size=10, shuffle=False)