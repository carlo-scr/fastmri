import torch
import torch.nn as nn


class UNet(nn.Module):
    """Simple U-Net for MRI reconstruction.

    Takes a 1-channel image (zero-filled reconstruction) and outputs
    a 1-channel reconstructed image.
    """

    def __init__(self, in_chans: int = 1, out_chans: int = 1, chans: int = 32, num_pool_layers: int = 4):
        super().__init__()

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Encoder
        ch = in_chans
        enc_chans = []
        for _ in range(num_pool_layers):
            self.down_layers.append(self._conv_block(ch, chans))
            enc_chans.append(chans)
            ch = chans
            chans *= 2

        # Bottleneck
        self.bottleneck = self._conv_block(ch, chans)

        # Decoder
        for i in range(num_pool_layers):
            self.up_layers.append(
                self._conv_block(chans + enc_chans[-(i + 1)], enc_chans[-(i + 1)])
            )
            chans = enc_chans[-(i + 1)]

        self.final = nn.Conv2d(chans, out_chans, kernel_size=1)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip_connections = []
        for down in self.down_layers:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder
        for i, up_layer in enumerate(self.up_layers):
            x = self.up(x)
            skip = skip_connections[-(i + 1)]
            # Handle size mismatch from odd dimensions
            if x.shape != skip.shape:
                x = nn.functional.pad(
                    x,
                    [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]],
                )
            x = torch.cat([x, skip], dim=1)
            x = up_layer(x)

        return self.final(x)
