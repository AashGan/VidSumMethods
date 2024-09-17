import torch.nn as nn 
class MLPM(nn.Module):
    def __init__(self,input_dims=1024,feedforward_dims=512,dropout=0.5) -> None:
        super().__init__()

        self.linear_net = nn.Sequential(
            nn.Linear(input_dims, feedforward_dims),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(feedforward_dims, input_dims)
        )  
        self.norm2 = nn.LayerNorm(input_dims)
        self.predictor =  nn.Sequential(nn.Linear(input_dims,1),nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        x_out = self.linear_net(x)
        x = x+self.dropout(x_out)
        x =  self.norm2(x)
        return self.predictor(x)