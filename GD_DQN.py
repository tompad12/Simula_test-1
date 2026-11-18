import torch
from torch import nn 
import matplotlib.pyplot as plt
import copy
device = "cuda" if torch.cuda.is_available() else "cpu"
class GD_DQN(nn.Module):
    def __init__(self, stack_size=4, num_actions=2, state_dim=4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(stack_size, 32, kernel_size=8, stride=4),  
            nn.SiLU(),
            # nn.BatchNorm2d(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.SiLU(),
        )
        self.fc_img=nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.SiLU()
        )
        self.fc_state=nn.Sequential(
            nn.Linear(state_dim,64),
            nn.SiLU(),
            nn.Linear(64,64),
            nn.SiLU()
        )
        self.head=nn.Sequential(
            nn.Linear(512+64,256),
            nn.SiLU(),
            nn.Linear(256,num_actions)
        )

    def forward(self,x,state):
        #B,T,H,W=x.shape
        img_feat=self.fc_img(self.conv(x))    
        state_feat=self.fc_state(state)       
        feat=torch.cat([img_feat,state_feat],dim=1)  
        return self.head(feat)


def crear_redes(device):
    online=GD_DQN().to(device)
    target=GD_DQN().to(device)
    d_x=torch.zeros(1,4,84,84,device=device)
    d_s=torch.zeros(1,4,device=device)
    online(d_x,d_s)
    target(d_x,d_s)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad=False
    return online,target
def update_target(online, target):
    target.load_state_dict(online.state_dict())

def crear_redes_dual(device):
    cube_online,cube_target=crear_redes(device)
    ship_online,ship_target=crear_redes(device)
    return cube_online,cube_target,ship_online,ship_target

def cargar_modelo_congelado(model,ruta,device):
    dummy_x=torch.zeros(1,4,84,84,device=device)
    dummy_s=torch.zeros(1,4,device=device)
    model(dummy_x,dummy_s)
    checkpoint=torch.load(ruta,map_location=device,weights_only=False)
    model.load_state_dict(checkpoint["online_state"])
    for p in model.parameters():
        p.requires_grad=False
    model.eval()
    return model