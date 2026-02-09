import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x, mask):
    mask = mask.unsqueeze(-1).float()
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class ANNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, dropout=0.3, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.fc1 = nn.Linear(embed_dim, 256)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)  # [B, L, D]
        pooled = masked_mean(x, attention_mask)
        x = F.relu(self.fc1(pooled))
        x = self.drop(x)
        return self.fc2(x)


class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, dropout=0.3, pad_id=0, kernel_size=5, num_filters=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)       # [B, L, D]
        x = x.transpose(1, 2)               # [B, D, L]
        x = F.relu(self.conv(x))            # [B, C, L]
        x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)  # [B, C]
        x = self.drop(x)
        return self.fc(x)


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, dropout=0.3, pad_id=0, hidden=128, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(embed_dim, hidden, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        out, _ = self.rnn(x)  # [B, L, H*dir]
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(out.size(0), 1, out.size(2))
        last = out.gather(1, idx).squeeze(1)
        last = self.drop(last)
        return self.fc(last)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, dropout=0.3, pad_id=0, hidden=128, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        out, _ = self.lstm(x)
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(out.size(0), 1, out.size(2))
        last = out.gather(1, idx).squeeze(1)
        last = self.drop(last)
        return self.fc(last)
