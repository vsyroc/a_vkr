import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchActivate(nn.Module):
    def __init__(self, num_features):
        super(BatchActivate, self).__init__()
        self.norm = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return F.relu(self.norm(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, stride=1, activation=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel, stride=stride, padding=padding)
        self.batchnorm = BatchActivate(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.batchnorm(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, stride=1):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel, padding, stride)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel, padding, stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, batch_activation=False):
        super(ResidualBlock, self).__init__()
        self.batch_activation = batch_activation
        self.norm = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = ConvBlock(in_channels, in_channels, kernel=3, stride=1, padding=1)
        self.conv2 = ConvBlock(in_channels, in_channels, kernel=3, stride=1, padding=1, activation=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x += residual
        # x = x.view(x.size(0),-1)

        if self.batch_activation:
            x = self.norm(x)

        return x


class UNetResNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, dropout=0.5, start_fm=16):
        super(UNetResNet, self).__init__()
        # Dropout
        self.drop = dropout
        # Pooling
        self.pool = nn.MaxPool2d((2, 2))

        # Encoder
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, start_fm, 3, padding=(1, 1)),
            ResidualBlock(start_fm),
            ResidualBlock(start_fm, batch_activation=True),
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(start_fm, start_fm * 2, 3, padding=(1, 1)),
            ResidualBlock(start_fm * 2),
            ResidualBlock(start_fm * 2, batch_activation=True),
        )

        self.encoder_3 = nn.Sequential(
            nn.Conv2d(start_fm * 2, start_fm * 4, 3, padding=(1, 1)),
            ResidualBlock(start_fm * 4),
            ResidualBlock(start_fm * 4, batch_activation=True),
        )

        self.encoder_4 = nn.Sequential(
            nn.Conv2d(start_fm * 4, start_fm * 8, 3, padding=(1, 1)),
            ResidualBlock(start_fm * 8),
            ResidualBlock(start_fm * 8, batch_activation=True),
        )

        self.middle = nn.Sequential(
            nn.Conv2d(start_fm * 8, start_fm * 16, 3, padding=3 // 2),
            ResidualBlock(start_fm * 16),
            ResidualBlock(start_fm * 16, batch_activation=True),
        )

        # Transpose conv
        self.deconv_4 = nn.ConvTranspose2d(start_fm * 16, start_fm * 8, kernel_size=(3, 3), stride=2, padding=1,
                                           output_padding=1)
        self.deconv_3 = nn.ConvTranspose2d(start_fm * 8, start_fm * 4, kernel_size=(3, 3), stride=2, padding=1,
                                           output_padding=1)
        self.deconv_2 = nn.ConvTranspose2d(start_fm * 4, start_fm * 2, kernel_size=(3, 3), stride=2, padding=1,
                                           output_padding=1)
        self.deconv_1 = nn.ConvTranspose2d(start_fm * 2, start_fm, kernel_size=(3, 3), stride=2, padding=1,
                                           output_padding=1)

        # Decoder
        self.decoder_4 = nn.Sequential(
            # nn.Dropout2d(dropout),
            nn.Conv2d(start_fm * 16, start_fm * 8, 3, padding=(1, 1)),
            ResidualBlock(start_fm * 8),
            ResidualBlock(start_fm * 8, batch_activation=True),
        )

        self.decoder_3 = nn.Sequential(
            # nn.Dropout2d(dropout),
            nn.Conv2d(start_fm * 8, start_fm * 4, 3, padding=(1, 1)),
            ResidualBlock(start_fm * 4),
            ResidualBlock(start_fm * 4, batch_activation=True),
        )

        self.decoder_2 = nn.Sequential(
            # nn.Dropout2d(dropout),
            nn.Conv2d(start_fm * 4, start_fm * 2, 3, padding=(1, 1)),
            ResidualBlock(start_fm * 2),
            ResidualBlock(start_fm * 2, batch_activation=True),
        )

        self.decoder_1 = nn.Sequential(
            # nn.Dropoutt2d(dropout),
            nn.Conv2d(start_fm * 2, start_fm, 3, padding=(1, 1)),
            ResidualBlock(start_fm),
            ResidualBlock(start_fm, batch_activation=True),
        )

        self.conv_last = nn.Conv2d(start_fm, n_classes, 1)

    def forward(self, x):
        # Encoder

        conv1 = self.encoder_1(x)  # 101
        x = self.pool(conv1)  # 50
        x = nn.Dropout2d(self.drop / 2)(x)

        conv2 = self.encoder_2(x)  # 50
        x = self.pool(conv2)  # 25
        x = nn.Dropout2d(self.drop)(x)

        conv3 = self.encoder_3(x)  # 25
        x = self.pool(conv3)  # 12
        x = nn.Dropout2d(self.drop)(x)

        conv4 = self.encoder_4(x)  # 12
        x = self.pool(conv4)  # 6
        x = nn.Dropout2d(self.drop)(x)

        # Middle
        x = self.middle(x)  # 6

        # Decoder
        x = self.deconv_4(x)  # 12
        x = torch.cat([conv4, x], dim=1)  # 16
        x = self.decoder_4(x)

        x = self.deconv_3(x)  # 25
        x = torch.cat([conv3, x], dim=1)
        x = self.decoder_3(x)

        x = self.deconv_2(x)  # 50
        x = torch.cat([conv2, x], dim=1)
        x = self.decoder_2(x)

        x = self.deconv_1(x)  # 101
        x = torch.cat([conv1, x], dim=1)
        x = self.decoder_1(x)

        out = (self.conv_last(x))  # 101
        return out
