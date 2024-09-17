import torch.nn as nn
class LSTMSum(nn.Module):
    def __init__(self,input_features=1024,dhidden=1024):
        super().__init__()
        
        self.dhidden = dhidden
        self.encoder = nn.LSTM(input_size = input_features,hidden_size = dhidden,batch_first=True) 
        self.ff1 = nn.Linear(in_features=self.dhidden, out_features=self.dhidden)
        self.ff2 = nn.Linear(in_features=self.dhidden, out_features=self.dhidden)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.layer_norm_ff1 = nn.LayerNorm(self.ff1.out_features)
        self.drop50 = nn.Dropout(0.5)
    def forward(self, x):
        y,_ = self.encoder(x) # residual output and layer norm is within the block

    
        # Frame level importance score regression
        # Two layer NN
        y = self.ff1(y)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ff1(y)

        y = self.ff2(y)
        y = self.sig(y)
        y = y.view(1, -1)

        return y 