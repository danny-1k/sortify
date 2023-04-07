from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.e_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 6), stride=2, padding=1)
        self.e_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.e_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.e_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1)
        self.e_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=1)

        self.d_1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=1)
        self.d_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=1)
        self.d_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.d_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.d_5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(4, 6), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        self.enc_2_latent = nn.Linear(256*10**2, 2048)
        self.latent_2_dec = nn.Linear(2048, 256*10**2)


    def encode(self, x):
        x = self.relu(self.e_1(x))
        x = self.relu(self.e_2(x))
        x = self.relu(self.e_3(x))
        x = self.relu(self.e_4(x))
        x = self.relu(self.e_5(x))
        x = x.view(x.shape[0], -1)
        x = self.enc_2_latent(x)

        return x

    def decode(self, x):
        x = self.latent_2_dec(x)
        x = x.view(x.shape[0], 256, 10, 10)
        x = self.relu(self.d_1(x))
        x = self.relu(self.d_2(x))
        x = self.relu(self.d_3(x))
        x = self.relu(self.d_4(x))
        x = self.d_5(x)

        return x

    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)

        return recon

