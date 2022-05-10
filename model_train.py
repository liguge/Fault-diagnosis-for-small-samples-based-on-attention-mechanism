import torch
import torch.nn as nn
import numpy as np
from datasave import train_loader, test_loader
from early_stopping import EarlyStopping
from label_smoothing import LSR
from oneD_Meta_ACON import MetaAconC
import time
from torchsummary import summary
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


#
setup_seed(1)


# class swish(nn.Module):
#     def __init__(self):
#         super(swish, self).__init__()
#
#     def forward(self, x):
#         x = x * F.sigmoid(x)
#         return x


# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # self.pool_w = nn.AdaptiveAvgPool1d(1)
        self.pool_w = nn.AdaptiveMaxPool1d(1)
        mip = max(6, inp // reduction)
        self.conv1 = nn.Conv1d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(mip, track_running_stats=False)
        self.act = MetaAconC(mip)
        self.conv_w = nn.Conv1d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, w = x.size()
        x_w = self.pool_w(x)
        y = torch.cat([identity, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_ww, x_c = torch.split(y, [w, 1], dim=2)
        a_w = self.conv_w(x_ww)
        a_w = a_w.sigmoid()
        out = identity * a_w
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.p1_1 = nn.Sequential(nn.Conv1d(1, 50, kernel_size=18, stride=2),
                                  nn.BatchNorm1d(50, track_running_stats=False),
                                  MetaAconC(50))
        self.p1_2 = nn.Sequential(nn.Conv1d(50, 30, kernel_size=10, stride=2),
                                  nn.BatchNorm1d(30, track_running_stats=False),
                                  MetaAconC(30))
        self.p1_3 = nn.MaxPool1d(2, 2)
        self.p2_1 = nn.Sequential(nn.Conv1d(1, 50, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(50, track_running_stats=False),
                                  MetaAconC(50))
        self.p2_2 = nn.Sequential(nn.Conv1d(50, 40, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(40, track_running_stats=False),
                                  MetaAconC(40))
        self.p2_3 = nn.MaxPool1d(2, 2)
        self.p2_4 = nn.Sequential(nn.Conv1d(40, 30, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(30, track_running_stats=False),
                                  MetaAconC(30))
        self.p2_5 = nn.Sequential(nn.Conv1d(30, 30, kernel_size=6, stride=2),
                                  nn.BatchNorm1d(30, track_running_stats=False),
                                  MetaAconC(30))
        self.p2_6 = nn.MaxPool1d(2, 2)
        self.p3_0 = CoordAtt(30, 30)
        self.p3_1 = nn.Sequential(nn.LSTM(124, 64, bidirectional=True))  #
        # self.p3_2 = nn.Sequential(nn.LSTM(128, 512))
        self.p3_3 = nn.Sequential(nn.AdaptiveAvgPool1d(1))
        self.p4 = nn.Sequential(nn.Linear(30, 10))

    def forward(self, x):
        p1 = self.p1_3(self.p1_2(self.p1_1(x)))
        p2 = self.p2_6(self.p2_5(self.p2_4(self.p2_3(self.p2_2(self.p2_1(x))))))
        encode = torch.mul(p1, p2)
        # p3 = self.p3_2(self.p3_1(encode))
        p3_0 = self.p3_0(encode).permute(1, 0, 2)
        p3_2, _ = self.p3_1(p3_0)
        # p3_2, _ = self.p3_2(p3_1)
        p3_11 = p3_2.permute(1, 0, 2)  # 
        p3_12 = self.p3_3(p3_11).squeeze()
        # p3_11 = h1.permute(1,0,2)
        # p3 = self.p3(encode)
        # p3 = p3.squeeze()
        # p4 = self.p4(p3_11)  # LSTM(seq_len, batch, input_size)
        # p4 = self.p4(encode)
        p4 = self.p4(p3_12)
        return p4


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
# model.load_state_dict(torch.load('./data7/B0503_AdamP_AMS_Nb.pt'))
# for m in model.modules():
#     if isinstance(m, nn.Conv1d):
#         #nn.init.normal_(m.weight)
#         #nn.init.xavier_normal_(m.weight)
#         nn.init.kaiming_normal_(m.weight)
#         #nn.init.constant_(m.bias, 0)
#     # elif isinstance(m, nn.GRU):
#     #     for param in m.parameters():
#     #         if len(param.shape) >= 2:
#     #             nn.init.orthogonal_(param.data)
#     #         else:
#     #             nn.init.normal_(param.data)
#     elif isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, mean=0, std=torch.sqrt(torch.tensor(1/30)))
# input = torch.rand(20, 1, 1024).to(device)
# output = model(input)
# print(output.size())
# with SummaryWriter(log_dir='logs', comment='Net') as w:
#      w.add_graph(model, (input,))
# tb = program.TensorBoard()
# tb.configure(argv=[None, '--logdir', 'logs'])
# url = tb.launch()
summary(model, input_size=(1, 1024))  
# criterion = nn.CrossEntropyLoss()

criterion = LSR()
# criterion = CrossEntropyLoss_LSR(device)
# from adabound import AdaBound
# optimizer = AdaBound(model.parameters(), lr=0.001, weight_decay=0.0001, amsbound=True)
# from EAdam import EAdam
# optimizer = EAdam(model.parameters(), lr=0.001, weight_decay=0.0001, amsgrad=True)
# optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.000, weight_decay=0.0001)
bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias')
others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias')
parameters = [{'parameters': bias_list, 'weight_decay': 0},
              {'parameters': others_list}]
# optimizer = Nadam(model.parameters())
# optimizer = RAdam(model.parameters())
# from torch_optimizer import AdamP
# from adamp import AdamP
from AdamP_amsgrad import AdamP
optimizer = AdamP(model.parameters(), lr=0.001, weight_decay=0.0001, nesterov=True, amsgrad=True)
# from adabelief_pytorch import AdaBelief
# optimizer = AdaBelief(model.parameters(), lr=0.001, weight_decay=0.0001, weight_decouple=True)
# from ranger_adabelief import RangerAdaBelief
# optimizer = RangerAdaBelief(model.parameters(), lr=0.001, weight_decay=0.0001, weight_decouple=True)
losses = []
acces = []
eval_losses = []
eval_acces = []
early_stopping = EarlyStopping(patience=10, verbose=True)
starttime = time.time()
for epoch in range(150):
        train_loss = 0
        train_acc = 0
        model.train()
        for img, label in train_loader:
            img = img.float()
            img = img.to(device)
            # label = (np.argmax(label, axis=1)+1).reshape(-1, 1)
            # label=label.float()

            label = label.to(device)
            label = label.long()
            out = model(img)
            out = torch.squeeze(out).float()
            # label=torch.squeeze(label)

            # out_1d = out.reshape(-1)
            # label_1d = label.reshape(-1)

            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(scheduler.get_lr())
            train_loss += loss.item()

            
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc

        losses.append(train_loss / len(train_loader))
        acces.append(train_acc / len(train_loader))
        #
        eval_loss = 0
        eval_acc = 0
        # net.eval() #
        model.eval()
        for img, label in test_loader:
            img = img.type(torch.FloatTensor)
            img = img.to(device)
            label = label.to(device)
            label = label.long()
            # img = img.view(img.size(0), -1)
            out = model(img)
            out = torch.squeeze(out).float()
            loss = criterion(out, label)
            # 
            eval_loss += loss.item()
            # 
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            # print(pred, '\n\n', label)
            acc = num_correct / img.shape[0]
            eval_acc += acc
        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))
        print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
              .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader),
                      eval_loss / len(test_loader), eval_acc / len(test_loader)))
        early_stopping(eval_loss / len(test_loader), model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
endtime = time.time()
dtime = endtime - starttime
print("time：%.8s s" % dtime)
torch.save(model.state_dict(), '\B0503_LSTM.pt')
import pandas as pd

pd.set_option('display.max_columns', None)  # 
pd.set_option('display.max_rows', None)  # 
