import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import AudioData
from tqdm import tqdm
from models import AutoEncoder


def setup():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"../runs/run_{timestamp}"
    save_dir = f"../checkpoints/{timestamp}"

    if not os.path.exists("../runs"):
        os.makedirs("../runs")

    if not os.path.exists("../checkpoints"):
        os.makedirs("../checkpoints")

    os.makedirs(run_name)
    os.makedirs(save_dir)

    return run_name, save_dir


def run(num_epochs, lr, batch_size, num_workers):


    run_name, save_dir = setup()

    writer = SummaryWriter(run_name)

    net = AutoEncoder()
    net.to("cuda")
    
    train = AudioData(train=True)
    test = AudioData(train=False)

    train = DataLoader(train, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    test = DataLoader(test, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    lossfn = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=lr)

    best_loss = float("inf")

    global_train_idx = 0
    global_test_idx = 0


    for epoch in tqdm(range(num_epochs)):

        current_loss = 0

        net.train()
        for x in tqdm(train):
            
            optimizer.zero_grad()

            x = torch.concat(x, dim=0)
            x = x.to("cuda").unsqueeze(1)

            p = net(x)

            loss = lossfn(p, x.clone())

            loss.backward()

            optimizer.step()

            writer.add_scalar("loss/train", loss.item(), global_train_idx)

            writer.add_image("images/train/original", x[-1].cpu(), global_train_idx)
            writer.add_image("images/train/reconstructed", p[-1].cpu(), global_train_idx)


            for name, param in net.named_parameters():
                writer.add_histogram(name, param, global_train_idx)
                writer.add_histogram(f"{name}.grad", param.grad, global_train_idx)


            global_train_idx += 1


        net.eval()
        with torch.no_grad():        
            for x in tqdm(test):

                x = torch.concat(x, dim=0)
                x = x.to("cuda").unsqueeze(1)

                p = net(x)

                loss = lossfn(p, x.clone()) 

                current_loss = (.4*current_loss)+(.6*loss.item())

                writer.add_scalar("loss/test", loss.item(), global_test_idx)

                writer.add_image("images/test/original", x[-1].cpu(), global_test_idx)
                writer.add_image("images/test/reconstructed", p[-1].cpu(), global_test_idx)

                for name, param in net.named_parameters():
                    writer.add_histogram(name, param, global_test_idx)
                    writer.add_histogram(f"{name}.grad", param.grad, global_test_idx)

                global_test_idx += 1

                if current_loss < best_loss:
                    checkpoint = {
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "loss": current_loss,
                    }

                    torch.save(checkpoint, f"{save_dir}/checkpoint.pt")

                    best_loss = current_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--workers", default=0, type=int)

    args = parser.parse_args()

    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size # real batch size will be `batch_size` * number of splits (9)
    num_workers = args.workers

    run(
        lr=lr,
        num_epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers
    )