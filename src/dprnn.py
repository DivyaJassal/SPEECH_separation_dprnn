import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, kernel_size=16, stride=8, out_channels=256):
        super().__init__()
        self.conv = nn.Conv1d(
            1,
            out_channels,
            kernel_size,
            stride=stride,
            bias=False
        )

    def forward(self, x):
        return F.relu(self.conv(x))


class Decoder(nn.Module):
    def __init__(self, kernel_size=16, stride=8, in_channels=256):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels,
            1,
            kernel_size,
            stride=stride,
            bias=False
        )

    def forward(self, x):
        return self.deconv(x)


def split_feature(x, chunk_size):

    batch, channels, length = x.shape

    hop = chunk_size // 2

    pad = chunk_size - (length % hop)

    if pad != chunk_size:
        x = F.pad(x, (0, pad))

    chunks = x.unfold(
        2,
        chunk_size,
        hop
    )

    chunks = chunks.permute(
        0,
        1,
        3,
        2
    )

    return chunks, length


def merge_feature(x, length):

    batch, channels, chunk, num_chunks = x.shape

    hop = chunk // 2

    output_length = hop * (num_chunks - 1) + chunk

    output = torch.zeros(
        batch,
        channels,
        output_length,
        device=x.device
    )

    count = torch.zeros_like(output)

    x = x.permute(
        0,
        1,
        3,
        2
    )

    for i in range(num_chunks):

        start = i * hop
        end = start + chunk

        output[:, :, start:end] += x[:, :, i, :]
        count[:, :, start:end] += 1


    output = output / count

    return output[:, :, :length]

class DPRNNBlock(nn.Module):
    def __init__(self, channels, hidden_size):
        super().__init__()

        self.intra_rnn = nn.LSTM(
            channels,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.intra_linear = nn.Linear(
            hidden_size * 2,
            channels
        )

        self.inter_rnn = nn.LSTM(
            channels,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.inter_linear = nn.Linear(
            hidden_size * 2,
            channels
        )

        self.norm1 = nn.GroupNorm(
            1,
            channels
        )

        self.norm2 = nn.GroupNorm(
            1,
            channels
        )

    def forward(self, x):
        batch, channels, chunk, num_chunks = x.shape

        intra = x.permute(0, 3, 2, 1)
        intra = intra.reshape(batch * num_chunks, chunk, channels)

        intra_out, _ = self.intra_rnn(intra)
        intra_out = self.intra_linear(intra_out)

        intra_out = intra_out.reshape(
            batch,
            num_chunks,
            chunk,
            channels
        )

        intra_out = intra_out.permute(0, 3, 2, 1)

        x = self.norm1(
            x + intra_out
        )

        inter = x.permute(0, 2, 3, 1)
        inter = inter.reshape(batch * chunk, num_chunks, channels)

        inter_out, _ = self.inter_rnn(inter)
        inter_out = self.inter_linear(inter_out)

        inter_out = inter_out.reshape(
            batch,
            chunk,
            num_chunks,
            channels
        )

        inter_out = inter_out.permute(0, 3, 1, 2)

        x = self.norm2(
            x + inter_out
        )

        return x


class MaskNet(nn.Module):
    def __init__(
        self,
        channels,
        hidden_size,
        num_sources,
        num_blocks=6,
        chunk_size=100
    ):
        super().__init__()

        self.chunk_size = chunk_size
        self.blocks = nn.ModuleList(
            [
                DPRNNBlock(
                    channels,
                    hidden_size
                )
                for _ in range(num_blocks)
            ]
        )

        self.output = nn.Conv1d(
            channels,
            channels * num_sources,
            1
        )

        self.num_sources = num_sources
        self.channels = channels

    def forward(self, x):
        batch, channels, length = x.shape

        x, rest = split_feature(
            x,
            self.chunk_size
        )

        for block in self.blocks:
            x = block(x)

        x = merge_feature(
            x,
            rest
        )

        masks = self.output(x)

        masks = masks.view(
            batch,
            self.num_sources,
            self.channels,
            -1
        )

        masks = F.softmax(
            masks,
            dim=1
        )

        return masks


class DPRNNTasNet(nn.Module):
    def __init__(
        self,
        num_sources=3,
        encoder_channels=256,
        hidden_size=128,
        kernel_size=16,
        stride=8,
        num_blocks=6,
        chunk_size=100
    ):
        super().__init__()

        self.encoder = Encoder(
            kernel_size,
            stride,
            encoder_channels
        )

        self.masknet = MaskNet(
            encoder_channels,
            hidden_size,
            num_sources,
            num_blocks,
            chunk_size
        )

        self.decoder = Decoder(
            kernel_size,
            stride,
            encoder_channels
        )

        self.num_sources = num_sources

    def forward(self, mixture):
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)

        encoded = self.encoder(
            mixture
        )

        masks = self.masknet(
            encoded
        )

        masked = encoded.unsqueeze(1) * masks

        batch, sources, channels, length = masked.shape

        masked = masked.reshape(
            batch * sources,
            channels,
            length
        )

        decoded = self.decoder(
            masked
        )

        decoded = decoded.reshape(
            batch,
            sources,
            -1
        )

        return decoded