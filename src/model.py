"""
Step 4: Lightweight 1D-CNN. Designed to stay under ~10-20k parameters,
comparable to the base paper's 9,220-parameter model.

Takes a 1D feature vector (FFT or wavelet energies, length N_BANDS) as input,
treated as a single-channel 1D signal.
"""
import torch
import torch.nn as nn


class LightweightCNN1D(nn.Module):
    def __init__(self, input_len=32, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pooling
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        # x shape: (batch, input_len) -> (batch, 1, input_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)  # (batch, 64)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LightweightCNN1D(input_len=32, n_classes=4)
    n_params = count_parameters(model)
    print(f"Total trainable parameters: {n_params:,}")
    dummy = torch.randn(8, 32)
    out = model(dummy)
    print(f"Output shape for batch of 8: {out.shape}")
