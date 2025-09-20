import torch
import torch.nn as nn

class CRNN(nn.Module):
    """
    CNN + BiLSTM (+ CTC log-probs)
    Input:  (B, 1, 48, 512) grayscale, 0..1
    Output: (B, T, num_classes) log-probs for CTC
    """
    def __init__(self, img_h: int = 48, num_channels: int = 1,
                 num_classes: int = 100, rnn_hidden: int = 256, rnn_layers: int = 2):
        super().__init__()

        # Feature extractor (keeps width long; compresses height)
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 48 -> 24

            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 24 -> 12

            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            # pool only in height to keep width resolution
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),  # 12 -> 6 (H)

            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),  # 6 -> 3 (H)

            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(inplace=True)  # 3 -> 2 (H)
        )

        # Sequence model over width dimension
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H=48, W=512)
        feats = self.cnn(x)             # (B, C=512, H', W')
        feats = feats.mean(2)           # average over height: (B, C, W')
        feats = feats.permute(0, 2, 1)  # (B, T=W', C=512)
        seq, _ = self.rnn(feats)        # (B, T, 2*hidden)
        logits = self.fc(seq)           # (B, T, num_classes)
        return self.log_softmax(logits) # CTC expects log-probs
