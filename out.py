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
import pandas as pd
import time

parser = argparse.ArgumentParser(description='PyTorch CallRecords Training Program')
parser.add_argument('--data', default='data', type=str, help='dataset directory')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--ckpt', default='ckpt.pth', type=str, help='checkpoint file path')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def pad_collate(batch):
    seq, time_diff, labels = zip(*batch)

    seq_padded = pad_sequence(seq, batch_first=True)
    time_diff_padded = pad_sequence(time_diff, batch_first=True)

    if labels[0] is not None:
        labels = torch.stack(labels)
    return seq_padded, time_diff_padded, labels

valid_set = CallRecords(root='data', train=False, valid=True)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

features_num = valid_set.features_num

print('==> Building model..')
print(f'Feature_num: {features_num}')
net = BiLSTMWithImprovedAttention(input_size=features_num,hidden_size=128,num_classes=2,attention_size=32,num_layer=2,dropout_rate=0.5, num_heads=4)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

temp = torch.load(args.ckpt)
net.load_state_dict(temp)
net.to(device)

def pred(epoch):
    global best_f1
    net.eval()
    test_loss = 0
    all_predictions = []
    precision = .0

    with torch.no_grad():
        for seq, time_diff, _ in valid_loader:
            seq, time_diff = seq.to(device), time_diff.to(device)
            outputs = net(seq)
            outputs = torch.softmax(outputs, dim=1)
            all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    pred = torch.tensor(all_predictions)

    valid_df = pd.read_csv('./data/CallRecords/raw/validationSet_res.csv')
    valid_seq_index_with_time_diff = valid_set.seq_index_with_time_diff
    msisdns = [valid_df['msisdn'][valid_seq_index_with_time_diff[i][0][0]] for i in range(len(valid_seq_index_with_time_diff))]
    msisdns = pd.Series(msisdns)

    # 验证预测结果排序和原始数据是否一致
    for i in range(len(valid_seq_index_with_time_diff)):
        assert len(valid_seq_index_with_time_diff[i][0]) == len(valid_df[valid_df['msisdn'] == msisdns[i]])

    res = pd.DataFrame()
    res['msisdn'] = msisdns
    res['is_sa'] = pred.cpu().numpy()
    res_name = time.strftime(r'%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    if not os.path.exists('./res'):
        os.makedirs('./res')
    res.to_csv('./res/' + res_name + '.csv', index=False)
    print("\n is_as 分布：")
    print(res['is_sa'].describe())

pred(0)