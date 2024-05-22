import torch
from torch import nn
from Parameters import lstm_input_size as input_size, lstm_hidden_size as hidden_size
from SoftClamp import SoftClamp


# Define LSTM model
class LSTM_MLP_Model(nn.Module):
    def __init__(self, input_size = input_size, hidden_size = hidden_size, num_layers=1):
        super(LSTM_MLP_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        # Output layer with the same number of features as the input
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_final = nn.Linear(128, 1)
        self.softClamp  = SoftClamp(5,2)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softClamp(self.fc_final(out))
        return out