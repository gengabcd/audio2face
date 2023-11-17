import torch
import torch.nn as nn
import torch.nn.functional as F


class NvidiaModel(nn.Module):
    def __init__(self, output_dim=52):
        super(NvidiaModel, self).__init__()
        self.output_dim = output_dim
        self.analysis_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1, out_channels=72, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            torch.nn.Conv2d(in_channels=72, out_channels=108, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            torch.nn.Conv2d(in_channels=108, out_channels=162, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            torch.nn.Conv2d(in_channels=162, out_channels=243, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            torch.nn.Conv2d(in_channels=243, out_channels=256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        ])

        self.articulation_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1), stride=(4, 1), padding=(1, 0)),
        ])
        self.linear1 = torch.nn.Linear(256, 150)
        self.dropout = torch.nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(150, self.output_dim)
        self.apply(lambda x: self.weights_init(x))

    ## x : (batch_size x ??? , 32, 64, 1)
    def forward(self, x):
        # print(x.shape)
        # x = x.view(x.shape[1], x.shape[4], x.shape[3], -1)
        x = torch.squeeze(x, 0)
        x = x.permute(0, 3, 2, 1)
        for layer in self.analysis_convs:
            # print(x.shape)
            x = layer(x)
            # x = F.leaky_relu(x,0.2)
            x = F.relu(x)
        for layer in self.articulation_convs:
            # print(x.shape)
            x = layer(x)
            # x = F.leaky_relu(x,0.2)
            x = F.relu(x)
        x = torch.squeeze(x)
        # x = self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        # x = torch.clamp(x,min=-1.0,max=1.0)
        return x

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, .0)
        elif isinstance(m, torch.nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        else:
            pass


def loss(output, label):
    loss_function1 = torch.nn.MSELoss(reduction='mean')
    loss_function2 = torch.nn.MSELoss(reduction='mean')
    loss_position = loss_function1(output, label)
    output_ = torch.diff(output, dim=0)
    label_ = torch.diff(label, dim=0)
    loss_motion = loss_function2(output_, label_)
    # loss = loss_position + 2*loss_motion
    loss_ = loss_position
    return loss_


if __name__ == '__main__':
    data = torch.rand((1, 20, 32, 64, 1))
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
