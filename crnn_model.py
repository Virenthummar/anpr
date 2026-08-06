import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulary: Digits 0-9, Uppercase A-Z, Blank symbol '-'
CHAR_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_CHAR = "-"
ALPHABET = BLANK_CHAR + CHAR_SET  # Index 0 is CTC Blank

class CRNN(nn.Module):
    """
    CRNN Architecture:
    1. CNN Backbone (Feature Extraction from image crop -> feature sequence)
    2. Map-to-Sequence Layer
    3. BiLSTM Sequence Modeling Layer
    4. CTC Linear Projection
    """
    def __init__(self, img_h=32, nc=1, nclass=len(ALPHABET), nh=256):
        super(CRNN, self).__init__()
        assert img_h % 16 == 0, "img_h must be a multiple of 16"

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Conv 1
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (64, 16, W/2)
            # Conv 2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (128, 8, W/4)
            # Conv 3 & 4
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # (256, 4, W/4)
            # Conv 5 & 6
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # (512, 2, W/4)
            # Conv 7
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)  # (512, 1, W/4)
        )

        # Recurrent BiLSTM Layers
        self.rnn = nn.Sequential(
            nn.LSTM(512, nh, bidirectional=True, batch_first=True),
        )
        self.rnn2 = nn.Sequential(
            nn.LSTM(nh * 2, nh, bidirectional=True, batch_first=True)
        )

        # Output Linear Projection
        self.linear = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        # x shape: (B, C, H, W)
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "Height after CNN must be 1"

        conv = conv.squeeze(2)  # (B, C, W)
        conv = conv.permute(0, 2, 1)  # (B, W, C) -> Batch, SequenceLength, FeatureDim

        recurrent, _ = self.rnn(conv)
        recurrent, _ = self.rnn2(recurrent)

        logits = self.linear(recurrent)  # (B, W, nclass)
        return logits

def ctc_decode(logits, alphabet=ALPHABET):
    """
    Greedy CTC Decoder:
    Collapses repeated characters and removes blank tokens (index 0).
    """
    probs = F.softmax(logits, dim=2)
    max_indices = torch.argmax(probs, dim=2)[0]  # Take first item in batch

    decoded_str = ""
    last_idx = 0
    conf_scores = []

    for i in range(len(max_indices)):
        idx = max_indices[i].item()
        conf = probs[0, i, idx].item()
        
        if idx != 0 and idx != last_idx:  # 0 is blank
            decoded_str += alphabet[idx]
            conf_scores.append(conf)
        last_idx = idx

    avg_conf = sum(conf_scores) / float(len(conf_scores)) if conf_scores else 0.0
    return decoded_str, round(avg_conf, 2)
