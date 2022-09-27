import torch
import torch.nn as nn
import numpy as np
from objectives import cca_loss


class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
		    		# nn.Tanh()
                )
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Dropout(0.1),
                    nn.ReLU()
                    
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, cca_size, hidden_size, input_size1):
        super(Decoder, self).__init__()
        self.layer_1_1 = nn.Linear(cca_size, 2 * hidden_size)
        self.layer_1_2 = nn.Linear(2 * hidden_size, input_size1)

    def forward(self, ccaoutput):
        rec = self.layer_1_2(self.layer_1_1(ccaoutput))
       
        return rec


class DeepCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DeepCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1).double()
        self.model2 = MlpNet(layer_sizes2, input_size2).double()

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss
        self.mseloss = torch.nn.MSELoss()

    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2
