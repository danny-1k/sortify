import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import AudioData
from tqdm import tqdm
from models import AutoEncoder


def run(num_epochs, lr, batch_size, run_name, num_workers):
    net = AutoEncoder()
    net.to("cuda")
    
    train = AudioData(train=True)
    test = AudioData(train=False)

    train = DataLoader(train, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    test = DataLoader(test, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    lossfn = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=lr)


    for epoch in tqdm(range(num_epochs)):

        for x in train:
            
            optimizer.zero_grad()

            x = torch.concat(x, dim=0)
            x = x.to("cuda").unsqueeze(1)

            print(x.shape)

            p = net(x)

            loss = lossfn(p, x.clone())

            print(loss.item())
            loss.backward()

            optimizer.step()

        with torch.no_grad():        
            for x in test:

                x = torch.concat(x, dim=0)
                x = x.to("cuda").unsqueeze(1)

                p = net(x)

                loss = lossfn(p, x.clone())

if __name__ == "__main__":

    epochs = 1
    lr = 1e-4
    batch_size = 3 # real batch size will be `batch_size` * number of splits (9)
    run(epochs, lr, batch_size, "001", 0)