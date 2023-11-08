import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta

class AudioEncoder(nn.Module):
    def __init__(self,latent_dim=128):
        super(AudioEncoder,self).__init__()

        self.melspec = ta.transforms.MelSpectrogram(sample_rate=16000,n_fft=2048,hop_length=160, n_mels=80)
        conv_len = 5
        self.conv_dimensions = torch.nn.Conv1d(80,128,kernel_size=conv_len)
        self.weights_init(self.conv_dimensions)

        convs = []
        for i in range(6):
            dilation = 2 * (i % 3 + 1)
            convs += [torch.nn.Conv1d(128, 128, kernel_size=conv_len, dilation=dilation)]
            self.weights_init(convs[-1])
        self.convs = torch.nn.ModuleList(convs)
        self.code = torch.nn.Linear(128, latent_dim)
        self.apply(lambda x: self.weights_init(x))

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            try:
                torch.nn.init.constant_(m.bias, .01)
            except:
                pass
    def forward(self,audio):
        B, T = audio.shape[0], audio.shape[1]
        x = self.melspec(audio).squeeze(1)
        x = torch.log(x.clamp(min=1e-10,max=None))
        if T == 1:
            x = x.unsqueeze(1)

        x = x.view(-1,x.shape[2],x.shape[3])
        x = F.leaky_relu(self.conv_dimensions(x), .2)

        for conv in self.convs:
            x_ = F.leaky_relu(conv(x), .2)
            if self.training:
                x_ = F.dropout(x_,.2)
                # print(x.shape)
                # print(x_.shape)
                l = (x.shape[2] - x_.shape[2]) // 2
                x = (x[:,:,l:-l] + x_) / 2
        # print(x.shape)
        x = torch.mean(x, dim=-1)
        # print(x.shape)
        x = x.view(B,T,x.shape[-1])
        x = self.code(x)
        return x
class ExpressionEncoder(nn.Module):
    def __init__(self, latent_dim = 128, n_blendshapes = 52):
        super(ExpressionEncoder,self).__init__()
        self.n_blenshapes = n_blendshapes
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(self.n_blenshapes,256),
            torch.nn.Linear(256,128)
        ])
        self.lstm = torch.nn.LSTM(input_size=128,hidden_size=128,num_layers=1,batch_first=True)
        self.code = torch.nn.Linear(128,latent_dim)
    def forward(self, blendshape):
        """
        :blendshape: B x T x n_blendshapes
        :return x: B x T x latent_dim
        """

        x = blendshape.sub(0.5)
        x = x.mul(2)

        for layer in self.layers:
            x = F.leaky_relu(layer(x),.2)
        x, _ = self.lstm(x)
        x = self.code(x)

        return x

class FusionMLP(nn.Module):
    def __init__(self,classes = 128, heads = 64, expression_dim = 128, audio_dim = 128):
        super(FusionMLP, self).__init__()
        self.classes = classes
        super.heads = heads











