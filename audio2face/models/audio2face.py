import torch
import torch.nn as nn
import torch.nn.functional as F

class NvidiaModel(nn.Module):
    def __init__(self, output_dim = 52):
        super(NvidiaModel, self).__init__()
        self.output_dim = 52
        self.analysis_convs = [
            torch.nn.Conv2d(in_channels=1, out_channels=72, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            torch.nn.Conv2d(in_channels=72, out_channels=108, kernel_size=(1, 3), stride=(1,2),padding=(0,1)),
            torch.nn.Conv2d(in_channels=108, out_channels=162, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            torch.nn.Conv2d(in_channels=162, out_channels=243, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            torch.nn.Conv2d(in_channels=243, out_channels=256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        ]

        self.articulation_convs = [
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1), stride=(4, 1), padding=(1, 0)),
        ]
        self.linear1 = torch.nn.Linear(256,150)
        self.dropout = torch.nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(150,self.output_dim)



    ##
    ## x : (??? , 32, 64, 1)
    def forward(self, x):
        x = x.view(x.shape[0], x.shape[3], x.shape[2], -1)
        for layer in self.analysis_convs:
            print(x.shape)
            x = layer(x)
            x = F.relu(x)
        for layer in self.articulation_convs:
            print(x.shape)
            x = layer(x)
            x = F.relu(x)
        x = x.view(x.shape[0],-1)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    def loss(self, output, label):
        loss_function = torch.nn.MSELoss(reduce = True, size_average=True)
        loss_position = loss_function(output, label)
        output_ = torch.diff(output,dim=0)
        label_ = torch.diff(label,dim=0)
        loss_motion = loss_function(output_,label_)
        loss = loss_position + loss_motion
        return loss







if __name__ == '__main__':
    data = torch.rand((20,32,64,1))
    # data = data.view(data.shape[0],data.shape[3],data.shape[2],-1)
    # print(data.shape)
    model = NvidiaModel()
    x = model(data)
    print(x.shape)
    # data = torch.rand((6,4))
    # data1 = torch.rand(6,4)
    # print(data)
    # print(data1)
    # res = loss(data,data1)
    # print(res)
    # print(res.shape)
