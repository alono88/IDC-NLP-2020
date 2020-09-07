import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.epochs_used = 0
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []    
        self.best_val_acc = 0

class BiLSTM(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lstm_dropout):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 2) 
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])
        return out

class AttentionModel(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lstm_dropout, sequence_length, alpha_dropout):
        super(AttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=lstm_dropout)
        self.alpha_net = nn.Linear(hidden_size*2, 1)
    
        self.dropout = nn.Dropout(alpha_dropout)
        self.fc = nn.Linear(hidden_size*2, num_classes)  
        self.input_size = input_size
        self.sequence_length = sequence_length
        
    def attention_layer(self, input_from_lstm):
        M = nn.Tanh()(input_from_lstm).permute(1,0,2)
        M = M.contiguous().view((-1, 2*self.input_size))
        wM = self.alpha_net(M)
        wM = wM.view((-1, self.sequence_length, 1)).squeeze()
        alpha_weights = F.softmax(wM, 1).unsqueeze(2) 
        r = torch.bmm(input_from_lstm.permute(1,2,0), alpha_weights).squeeze()
        return r        
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        lstm_out, (last_hidden_state, last_cell_state) = self.lstm(x, (h0, c0)) 
        attention_out = self.attention_layer(lstm_out.permute(1,0,2))
        h_star = nn.Tanh()(attention_out)
        out = self.fc(self.dropout((h_star)))
        return out