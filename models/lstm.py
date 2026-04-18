import torch.nn as nn

class AlgorithmicMomentumLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2):
        super(AlgorithmicMomentumLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.5
        )
        
        self.fc = nn.Linear(hidden_size, 3)  # 3 classes: up, down, neutral

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden_state = out[:, -1, :]
        logits = self.fc(last_hidden_state)

        return logits