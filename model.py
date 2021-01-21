import torch
import torch.nn as nn
from torch.autograd import Variable
from MobileNetV2 import MobileNetV2
import torchvision.models as models


class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, device, three, bidirectional=True, dropout=True):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device
        self.three = three


        # #モデルの読み込み
        net = MobileNetV2(width_mult=width_mult)
        state_dict_mobilenet = torch.load('mobilenet_v2.pth.tar')
        if pretrain:
            net.load_state_dict(state_dict_mobilenet,strict=False)

        self.cnn = nn.Sequential(*list(net.children())[0][:19])  ##self.feature





        self.rnn = nn.LSTM(int(1280*width_mult if width_mult > 1.0 else 1280),
                           self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)
        if self.three == True:
            if self.bidirectional:
                self.lin = nn.Linear(2*self.lstm_hidden, 3)
            else:
                self.lin = nn.Linear(self.lstm_hidden, 3)


        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).to(self.device),torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).to(self.device)
        else:
            return torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(self.device),torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(self.device)



    # def forward(self, x, lengths=None):
    def forward(self,x):
        batch_size, timesteps, C, H, W = x.size()  ##torch.Size([8, 300, 3, 224, 224])
        self.hidden = self.init_hidden(batch_size)

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)  ##torch.Size([2400, 3, 224, 224])
        c_out = self.cnn(c_in)  ##torch.Size([2400, 1280, 7, 7])##特徴マップ

        c_out = c_out.mean(3).mean(2)  ##torch.Size([2400, 1280])  ##Global average pooling


        if self.dropout:
            c_out = self.drop(c_out)


        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)  ##torch.Size([8, 300, 1280])
        r_out, states = self.rnn(r_in, self.hidden)  ##r_out:torch.Size([8, 300, 512]),  len(states)=2
        out = self.lin(r_out)  ##torch.Size([8, 300, 12])
        # out.shape => torch.Size([1, 300, 13])
        if self.three == True:
            out = out.view(batch_size*timesteps, 3)
        # out.shape => torch.Size([300, 13])
        return out,


