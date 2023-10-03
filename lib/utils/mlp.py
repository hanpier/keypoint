import functools
import torch.nn as nn



class MLP(nn.Module):

    def __init__(
        self,
        in_features=1024, # 4400 56320
        hidden_layers=[1024, 1024],
        activation="leaky_relu",
        bn=True,
        dropout=0.0,
    ):
        super().__init__()
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]

        assert len(hidden_layers) > 0
        self.out_features = hidden_layers[-1]

        mlp = []

        if activation == "relu":
            act_fn = functools.partial(nn.ReLU, inplace=True)
        elif activation == "leaky_relu":
            act_fn = functools.partial(nn.LeakyReLU, inplace=True)
        else:
            raise NotImplementedError

        for hidden_dim in hidden_layers:
            mlp += [nn.Linear(in_features, hidden_dim)]
            if bn:
                mlp += [nn.BatchNorm1d(hidden_dim)]
            mlp += [act_fn()]
            if dropout > 0:
                mlp += [nn.Dropout(dropout)]
            in_features = hidden_dim
        self.feat_conv = nn.Conv2d(256, 1024, kernel_size=(11, 20), padding=0)
        self.mlp = nn.Sequential(*mlp)
        self.mlp = nn.Sequential(*mlp, nn.Linear(in_features, 1))

    def forward(self, x):
        x = self.feat_conv(x).view(x.shape[0], -1)
        x = self.mlp(x)
        return x


