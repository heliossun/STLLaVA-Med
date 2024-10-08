import torch
import torch.nn as nn
import re
import torch.nn.functional as F
from collections import OrderedDict
from .cluster import Clustering
class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
    
def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'cluster')
    print("projector type:",projector_type)
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    if projector_type == "cluster":
        return Clustering(config.mm_hidden_size, config.hidden_size,256, 2)
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

if __name__ == "__main__":
    m = Clustering(1024,512,4)
    print(list(m.modules()))
    input = torch.randn(2, 512,1024)
    out = m(input)
    # cluster = Clustering(3,96,10,10,0,0,3,32)
    # out = cluster(input)
    #print(out.shape)