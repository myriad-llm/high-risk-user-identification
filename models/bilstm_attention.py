import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.query = nn.Linear(input_size, hidden_size * num_heads)
        self.key = nn.Linear(input_size, hidden_size * num_heads)
        self.value = nn.Linear(input_size, hidden_size * num_heads)

        self.fc_out = nn.Linear(hidden_size * num_heads, hidden_size * 8)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 分割头
        Q = Q.view(batch_size, -1, self.num_heads, self.hidden_size).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.hidden_size).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.hidden_size).transpose(1, 2)

        # 计算注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 加权求和
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size * self.num_heads)
        
        output = self.fc_out(attention_output)

        return output

class BiLSTMWithImprovedAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, attention_size, num_layer, dropout_rate, num_heads):
        super(BiLSTMWithImprovedAttention, self).__init__()

        # 双向LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layer,
                            bidirectional=True, batch_first=True,
                            dropout=dropout_rate)

        # 多头注意力机制
        self.attention = MultiHeadAttention(hidden_size * 2, attention_size, num_heads)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # LSTM层
        lstm_out, _ = self.lstm(x)

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # 多头注意力机制
        attention_output = self.attention(lstm_out)
        attention_output = attention_output[:, -1, :] 
        output = self.fc(attention_output)
        return output
