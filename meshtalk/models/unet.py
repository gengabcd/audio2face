import torch
import torch.nn.functional as F

class Unet(torch.nn.Module):
    def __init__(self, template, classes = 128,heads = 64,n_blendshapes = 52):
        super(Unet,self).__init__()
        self.classes = classes
        self.heads = heads
        self.n_blendshapes = n_blendshapes
        self.template = template

        self.encoder = torch.nn.ModuleList([
            torch.nn.Linear(n_blendshapes,512),
            torch.nn.Linear(512,256),
            torch.nn.Linear(256,128)
        ])

        self.fusion = torch.nn.Linear(heads*classes + 128, 128)
        self.temporal = torch.nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
        self.decoder = torch.nn.ModuleList([
            torch.nn.Linear(128,256),
            torch.nn.Linear(256,512),
            torch.nn.Linear(512,n_blendshapes)
        ])
        self.vertex_bias = torch.nn.Parameter(torch.zeros(n_blendshapes))
    def _encode(self, x):
        skips = []
        for i, layer in enumerate(self.encoder):
            skips = [x] + skips
            x = F.leaky_relu(layer(x),0.2)
        return x, skips
    def _fuse(self,geom_encoding, expression_encoding):
        expression_encoding = expression_encoding.view(expression_encoding.shape[0],expression_encoding.shape[1], self.heads*self.classes)
        x = self.fusion(torch.cat([geom_encoding,expression_encoding],dim=-1))
        x = F.leaky_relu(x,.2)
        return x
    def _decode(self,x,skips):
        x, _ = self.temporal(x)
        for i, layer in enumerate(self.decoder):
            x = skips[i] + F.leaky_relu(layer(x), 0.2)
        return x
    def forward(self, expression_encoding):
        x = self.template.sub(0.5)
        x = x.mul(2.0)

        geom_encoding, skips = self._encode(x)
        x = self._fuse(geom_encoding,expression_encoding)
        x = self._decode(x,skips)
        x.view(x.shape[0], x.shape[1],self.n_blendshapes)
        x = x.mul(0.5)
        x = x.sub(-0.5)

        return x
