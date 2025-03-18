import torch.nn as nn
import torch.nn.functional as F


class WaveformEncoder(nn.Module):
    def __init__(self, n_input, n_output, stride=160, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=400, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.lstm = nn.LSTM(n_channel, n_channel, num_layers=2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = x.transpose(-1, -2)
        x, _ = self.lstm(x)
        x = x.transpose(-1, -2)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.transpose(-1, -2)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)
