import torch
from torch import nn

class AutoEncoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder_head = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,3), stride=2, padding=1)
		
		self.decoder_head = nn.Sequential(
			nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(4, 6), stride=2, padding=1),
			nn.ELU(),
			nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
			nn.Sigmoid()
		)

		self.encoder_body = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
			nn.ELU(),
			nn.MaxPool2d(2,2),

			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
			nn.ELU(),
			nn.MaxPool2d(2,2),

			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
			nn.ELU(),
			nn.MaxPool2d(2,2),

			nn.Flatten()
		)

		self.decoder_body = nn.Sequential(
			nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
			nn.ELU(),

			nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
			nn.ELU(),

			nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=2, stride=2),
			nn.ELU(),
		)

		self.to_latent = nn.Linear(128*8*8, 512)
		self.from_latent = nn.Linear(512, 128*8*8)


	def forward(self, x):
		x = self.encode(x)
		x = self.decode(x)
		
		return x		
	
	def encode(self, x):
		x = self.encoder_head(x)
		x = self.encoder_body(x)
		x = self.to_latent(x)
		return x
	
	def decode(self, latent):
		x = self.from_latent(latent)
		x = x.view(x.shape[0], 128, 8, 8)
		x = self.decoder_body(x)
		x = self.decoder_head(x)
		return x