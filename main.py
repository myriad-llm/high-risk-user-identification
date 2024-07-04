import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import BiLSTMWithImprovedAttention
from utils.dataset import CallRecords

parser = argparse.ArgumentParser(description='PyTorch CallRecords Training Program')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--data', default='data', type=str, help='dataset directory')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--epoch', default=100, type=int, help='batch size')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
best_f1 = 0
start_epoch = 0


def pad_collate(batch):
    seq, time_diff, labels = zip(*batch)

    seq_padded = pad_sequence(seq, batch_first=True)
    time_diff_padded = pad_sequence(time_diff, batch_first=True)
    labels = torch.stack(labels)
    return seq_padded, time_diff_padded, labels


train_set = CallRecords(root=args.data, train=True)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)

test_set = CallRecords(root=args.data, train=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

assert train_set.features_num == test_set.features_num, 'Train and Test set features are not equal'
features_num = train_set.features_num

print('==> Building model..')
print(f'Feature_num: {features_num}')
net = BiLSTMWithImprovedAttention(input_size=features_num, hidden_size=128, num_classes=2, attention_size=32, num_layer=2, dropout_rate=0.5, num_heads=4)
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
    best_f1 = checkpoint['f1']
    start_epoch = checkpoint['epoch']

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)


def train():
    net.train()
    train_loss = 0
    all_predictions = []
    all_labels = []

    for seq, time_diff, labels in tqdm(train_loader, desc='Training', leave=False):
        seq, time_diff, labels = seq.to(device), time_diff.to(device), labels.to(device)
        labels = torch.argmax(labels, dim=1)

        optimizer.zero_grad()
        outputs = net(seq)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        outputs = torch.softmax(outputs, dim=1)
        all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average='macro',
        warn_for=tuple(),
    )
    acc = sum([1 for i, j in zip(all_predictions, all_labels) if i == j]) / len(all_predictions)
    return train_loss / len(train_loader), precision, recall, f1, acc


def test():
    net.eval()
    test_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for seq, time_diff, labels in tqdm(test_loader, desc='Testing', leave=False):
            seq, time_diff, labels = seq.to(device), time_diff.to(device), labels.to(device)
            outputs = net(seq)
            labels = torch.argmax(labels, dim=1)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()

            outputs = torch.softmax(outputs, dim=1)
            all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average='macro',
        warn_for=tuple(),
    )
    acc = sum([1 for i, j in zip(all_predictions, all_labels) if i == j]) / len(all_predictions)
    return test_loss / len(test_loader), precision, recall, f1, acc


pbar = tqdm(range(start_epoch, start_epoch + args.epoch), total=args.epoch)
for epoch in pbar:
    train_loss, train_precision, train_recall, train_f1, train_acc = train()
    test_loss, test_precision, test_recall, test_f1, test_acc = test()

    info = f'''
Epoch: {epoch + 1} / {args.epoch}, \
Loss: {train_loss:.4f} / {test_loss:.4f}, \
Prec: {train_precision:.4f} / {test_precision:.4f}, \
Recall: {train_recall:.4f} / {test_recall:.4f}, \
F1: {train_f1:.4f} / {test_f1:.4f}, \
Acc: {train_acc:.4f} / {test_acc:.4f}'''

    # Save checkpoint.
    if test_f1 > best_f1:
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        state = {
            'net': net.state_dict(),
            'f1': test_f1,
            'epoch': epoch,
        }
        torch.save(state, './checkpoint/ckpt.pth')

        best_f1 = test_f1
        info += ' [Saved]'

    pbar.write(info)
