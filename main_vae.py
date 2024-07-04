import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import VAE
from utils.dataset import CallRecords

parser = argparse.ArgumentParser(description='PyTorch CallRecords Training Program')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--data', default='data', type=str, help='dataset directory')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--epoch', default=100, type=int, help='batch size')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
best_loss = sys.float_info.max
start_epoch = 0

train_set = CallRecords(root=args.data, train=True, non_seq=True)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

test_set = CallRecords(root=args.data, train=False, non_seq=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

assert train_set.features_num == test_set.features_num, 'Train and Test set features are not equal'
features_num = train_set.features_num

print('==> Building model..')
print(f'Feature_num: {features_num}')
net = VAE(in_channels=features_num, latent_dim=128)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']


def loss_fn(recon_x, x, mu, log_var):
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kld


optimizer = optim.Adam(net.parameters(), lr=args.lr)


def train():
    net.train()
    total_loss = 0

    for records in tqdm(train_loader, desc='Training', leave=False):
        records = records.to(device)

        pred, mu, sigma = net(records)
        loss = loss_fn(pred, records, mu, sigma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() / len(records)

    return total_loss / len(train_loader)


pbar = tqdm(range(start_epoch, start_epoch + args.epoch), total=args.epoch)
for epoch in pbar:
    loss = train()

    info = f'''Epoch: {epoch + 1} / {args.epoch}, Loss: {loss:.4f}'''

    # Save checkpoint.
    if loss < best_loss:
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        state = {
            'net': net.state_dict(),
            'loss': loss,
            'epoch': epoch,
        }
        torch.save(state, './checkpoint/ckpt.pth')

        best_loss = loss
        info += ' [Saved]'

    pbar.write(info)
