"""
Step 4: Lightweight 1D-CNN. Designed to stay under ~10-20k parameters,
comparable to the base paper's 9,220-parameter model.

Takes a 1D feature vector (FFT or wavelet energies, length N_BANDS) as input,
treated as a single-channel 1D signal.
"""
import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub


class LightweightCNN1D(nn.Module):
    def __init__(self, input_len=32, n_classes=4):
        super().__init__()
        # QuantStub/DeQuantStub are transparent pass-throughs in FP32; they only
        # become real quant/dequant ops after prepare()/convert() for static
        # quantization. They add no parameters, so FP32 checkpoints load as-is.
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
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
        x = self.quant(x)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)  # (batch, 64)
        x = self.classifier(x)
        return self.dequant(x)

    def fuse_model(self):
        """Fuse each Conv+BN+ReLU triplet in `features` into a single op.
        Required before static quantization: it folds BatchNorm into the Conv
        weights and lets the quantized kernel emit one INT8 Conv-ReLU. Call in
        eval() mode, and only on a freshly loaded FP32 model (fusion rewrites
        the module tree, so it must happen after load_state_dict)."""
        torch.ao.quantization.fuse_modules(
            self,
            [
                ["features.0", "features.1", "features.2"],
                ["features.3", "features.4", "features.5"],
                ["features.6", "features.7", "features.8"],
            ],
            inplace=True,
        )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LightweightCNN1D(input_len=32, n_classes=4)
    n_params = count_parameters(model)
    print(f"Total trainable parameters: {n_params:,}")
    dummy = torch.randn(8, 32)
    out = model(dummy)
    print(f"Output shape for batch of 8: {out.shape}")
