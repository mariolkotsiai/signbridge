import torch
import torch.nn as nn


class SignLanguageLSTM(nn.Module):
    """
    LSTM classifier for sign language landmark sequences.

    Input shape:  (batch_size, seq_len, input_size)
        - seq_len:    number of frames per sample (π.χ. 30)
        - input_size: αριθμός landmark features ανά frame (π.χ. 126
                      = 2 χέρια * 21 σημεία * 3 συντεταγμένες)

    Output shape: (batch_size, num_classes) — raw logits (χωρίς softmax,
                  γιατί χρησιμοποιείται με nn.CrossEntropyLoss).
    """

    def __init__(self, num_classes, input_size=126, hidden_size=128,
                 num_layers=2, dropout=0.3):
        super(SignLanguageLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 λόγω bidirectional

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Παίρνουμε το τελευταίο hidden state και από τις 2 κατευθύνσεις
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        last_forward = h_n[-2]   # τελευταίο layer, forward direction
        last_backward = h_n[-1]  # τελευταίο layer, backward direction
        combined = torch.cat((last_forward, last_backward), dim=1)  # (batch, hidden_size*2)

        out = self.dropout(combined)
        out = self.fc(out)  # (batch, num_classes)
        return out