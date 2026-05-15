from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAtt2d(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()

        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        x_norm = self.norm(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out

        return x_seq.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

class UNet(nn.Module):

    def __init__(self, channels: List[int] = [32, 64, 128], convs_per_level=2,
                 kernel_size=3, pool_size=2, padding=1, num_heads_att=4):
        super().__init__()

        full_channels = [1] + channels + channels[-2::-1]
        self.mid = len(full_channels) // 2

        self.encoder_convs = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(
                    full_channels[i] if j == 0 else full_channels[i+1],
                    full_channels[i+1], kernel_size, padding=padding
                )
                for j in range(convs_per_level)
            ])
            for i in range(self.mid)
        ])

        self.decoder_convs = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(
                    (full_channels[self.mid + i] + full_channels[self.mid - i - 1]) if j == 0
                    else full_channels[self.mid + i + 1],
                    full_channels[self.mid + i + 1],
                    kernel_size, padding=padding
                )
                for j in range(convs_per_level)
            ])
            for i in range(len(full_channels) - self.mid - 1)
        ])

        #Only uses attention when the number of heads is valid.
        bottleneck_channels = full_channels[self.mid]
    
        if num_heads_att < 2:
            self.attention = nn.Identity()
        else:
            assert bottleneck_channels % num_heads_att == 0, (
                f"number of heads for attention is invalid"
            )
            self.attention = SelfAtt2d(bottleneck_channels, num_heads=num_heads_att)

        self.output_conv = nn.Conv2d(full_channels[-1], 1, kernel_size=1)

        self.pool = nn.MaxPool2d(pool_size)
        self.upsample = nn.Upsample(scale_factor=pool_size)
        self.activation_fn = F.relu

        self.config = {
            'channels':        channels,
            'convs_per_level': convs_per_level,
            'kernel_size':     kernel_size,
            'pool_size':       pool_size,
            'padding':         padding,
            'num_heads_att':   num_heads_att,
        }

    def forward(self, x):
        skips = []
        last_encoder = len(self.encoder_convs) - 1

        for i, conv_block in enumerate(self.encoder_convs):
            for conv in conv_block:
                x = self.activation_fn(conv(x))

            if i < last_encoder:
                skips.append(x)
                x = self.pool(x)

        x = self.attention(x)

        for i, conv_block in enumerate(self.decoder_convs):
            x = self.upsample(x)
            x = torch.cat([x, skips.pop()], dim=1)

            for conv in conv_block:
                x = self.activation_fn(conv(x))

        return self.output_conv(x)