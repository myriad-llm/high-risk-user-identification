import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, attention_size, num_layer, dropout_rate):
        super(BiLSTMWithAttention, self).__init__()

        # 双向LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layer,
                            bidirectional=True, batch_first=True,
                            dropout=dropout_rate)

        # 注意力机制的参数
        self.attention_W = nn.Linear(hidden_size * 2, attention_size)
        self.attention_U = nn.Linear(attention_size, 1, bias=False)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x = x.unsqueeze(1)  # 添加一个通道维度
        # LSTM层
        lstm_out, _ = self.lstm(x)

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # 注意力机制
        attention_score = torch.tanh(self.attention_W(lstm_out))
        attention_weight = F.softmax(self.attention_U(attention_score), dim=1)
        attention_output = torch.sum(attention_weight * lstm_out, dim=1)

        # 全连接层
        output = self.fc(attention_output)

        return output