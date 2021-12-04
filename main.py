from torch import optim

from datautil import *
import random
import time
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torch.backends.cudnn
from tqdm import tqdm
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def train(
        model : nn.Module, epochs : int,
        train_loader : DataLoader, validation_loader,
        loss_func,
        optimizer,
        scheduler,
        save_path,
        plot
):
    model.train()
    torch.cuda.empty_cache()
    start = time.time()
    loss_values = []
    accr_values = []
    loss_valids = []
    accr_valids = []
    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = 0
        print(f"EPOCH {epoch+1} training begins...")
        correct, total = 0, 0
        for i, data in enumerate(train_loader):
            fp, label = data
            fp = fp.to(device)
            label = label.float().to(device)
            output = model(fp)
            loss = loss_func(output, label)
            train_loss += loss.item()
            pred = output.round()
            correct += pred.eq(label.view_as(pred)).sum().item()
            total += label.size(0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = train_loss/len(train_loader)
        epoch_accr = 100*correct/total
        loss_values.append(epoch_loss)
        accr_values.append(epoch_accr)
        print(f"Train epoch {epoch+1} / {epochs}",
              f"Loss {epoch_loss:.4f}, Accuracy {epoch_accr:.2f}%",
              f"Training Time {(time.time()-epoch_start)/60:.2f} min")
        if validation_loader is not None:
            model.eval()
            valid_loss = 0
            valid_correct, valid_total = 0, 0
            with torch.no_grad():
                for i, data in enumerate(validation_loader):
                    fp, label = data
                    fp = fp.to(device)
                    label = label.float().to(device)
                    output = model(fp)
                    valid_loss += loss_func(output, label).item()
                    pred = output.round()
                    valid_correct += pred.eq(label.view_as(pred)).sum().item()
                    valid_total += label.size(0)
            valid_loss = valid_loss/len(validation_loader)
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()
            valid_accr = 100*valid_correct/valid_total
            loss_valids.append(valid_loss)
            accr_valids.append(valid_accr)
            if max(accr_valids) == valid_accr and save_path is not None:
                torch.save(model.state_dict(), save_path)
                print(f"saved model to {save_path}")
            print(f"Validation epoch {epoch+1} / {epochs}",
                  f"Loss {valid_loss:.4f}, Accuracy {valid_accr:.2f}%")
        print()
    print(f"Total training time {(time.time()-start)/60:.2f} minutes taken")
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.plot(loss_values)
        ax1.plot(loss_valids)
        ax1.set_title("Loss")
        ax2.plot(accr_values)
        ax2.plot(accr_valids)
        ax2.set_title("Accuracy (%)")
        plt.show()

def svm_loss(output, y):
    return torch.mean(torch.clamp(1 - y * output, min=0))

def main():
    train_loader, test_loader = load_DILI_data()
    model = MLP(in_ch = 881)
    model.to(device)
    summary(model, (1, 881))
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    train(
        model=model, epochs=50,
        train_loader=train_loader,
        validation_loader=test_loader,
        optimizer=optimizer,
        loss_func=nn.MSELoss(),
        scheduler=None,
        save_path=None,
        plot=True
    )
main()