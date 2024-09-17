import torch
import math

def getPositionEncoding(seq_len, d, n=100000):
    pe = torch.zeros(seq_len, d)    
    # create position column   
    k = torch.arange(0, seq_len).unsqueeze(1)  

    # calc divisor for positional encoding 
    div_term = torch.exp(                                 
            torch.arange(0, d, 2) * -(math.log(n) / d)
    )

    # calc sine on even indices
    pe[:, 0::2] = torch.sin(k * div_term)    

    # calc cosine on odd indices   
    pe[:, 1::2] = torch.cos(k * div_term)
    pe = pe.unsqueeze(0)
  
    return pe