import config
from utils import dataset, loss, metrics, common
from models import UNet
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

def val(model, val_loader, loss_func, n_labels, device):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)

            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, target)

    return val_loss.avg, val_dice.avg


def train(model, train_loader, optimizer, loss_func, n_labels, alpha, epoch, device):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.float(), target.long()
        target = common.to_one_hot_3d(target, n_labels)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(data)
        losses = [loss_func(output, target) for output in outputs[:-1]]  # Assuming outputs[:-1] are your intermediate outputs
        main_loss = loss_func(outputs[-1], target)  # Assuming outputs[-1] is your final output

        loss = main_loss + alpha * sum(losses)
        loss.backward()
        optimizer.step()

        train_loss.update(main_loss.item(), data.size(0))
        train_dice.update(outputs[-1], target)

    return train_loss.avg, train_dice.avg


if __name__ == '__main__':
    args = config.args
    device = torch.device(args.device)
    save_path = os.path.join('./experiments', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    
    # Initialize log file
    log_file = os.path.join(save_path, 'log.txt')
    with open(log_file, 'w') as f:
        f.write('Epoch,Train_Loss,Train_Dice,Val_Loss,Val_Dice\n')
    
    train_loader = DataLoader(dataset=dataset.CTDataset(args), batch_size=args.batch_size, num_workers=args.n_threads, shuffle=True)
    val_loader = DataLoader(dataset=dataset.CTDataset(args), batch_size=1, num_workers=args.n_threads, shuffle=False)
    
    model = UNet.UNet(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = loss.CombinedLoss()
    n_labels = args.n_labels
    alpha = 0.4  # 深监督衰减系数初始值
    
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    
    for epoch in range(args.epochs):
        train_loss, train_dice = train(model, train_loader, optimizer, loss_func, n_labels, alpha, epoch, device)
        val_loss, val_dice = val(model, val_loader, loss_func, n_labels, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        
        # Write log
        with open(log_file, 'a') as f:
            f.write(f'{epoch},{train_loss},{train_dice},{val_loss},{val_dice}\n')
    
    # Plot and save the loss and dice graphs
    epochs = range(args.epochs)
    
    plt.figure()
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    
    plt.figure()
    plt.plot(epochs, train_dices, 'b', label='Training Dice')
    plt.plot(epochs, val_dices, 'r', label='Validation Dice')
    plt.title('Training and validation Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'dice_plot.png'))
