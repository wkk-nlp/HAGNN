import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
import torch
import torch.nn as nn
from HiGraph import HAGNN
from utils import build_dataset
from dataLoader import CreateGraph
from sklearn import metrics
import time

device = torch.device('cuda')

def save_model(dic, save_file):
    with open(save_file, 'wb') as f:
        torch.save(dic, f)


parser = argparse.ArgumentParser(description='HAGNN Model')

# Hyperparameters
parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs [default: 20]')
parser.add_argument('--n_iter', type=int, default=2, help='iteration hop [default: 1]')

parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 300]')
parser.add_argument('--feat_embed_size', type=int, default=50,
                    help='feature embedding size [default: 50]')
parser.add_argument('--n_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state [default: 128]')
parser.add_argument('--lstm_layers', type=int, default=2, help='Number of lstm layers111 [default: 2]')
parser.add_argument('--bidirectional', action='store_true', default=True,
                    help='whether to use bidirectional LSTM [default: True]')
parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature [default: 128]')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden size [default: 64]')
parser.add_argument('--n_aspect', type=int, default=15, help='number of aspects [default: 15]')
parser.add_argument('--ffn_inner_hidden_size', type=int, default=64,
                    help='PositionwiseFeedForward inner hidden size [default: 512]')
parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1,
                    help='recurrent dropout prob [default: 0.1]')
parser.add_argument('--atten_dropout_prob', type=float, default=0.1, help='attention dropout prob [default: 0.1]')
parser.add_argument('--ffn_dropout_prob', type=float, default=0.1,
                    help='PositionwiseFeedForward dropout prob [default: 0.1]')
parser.add_argument('--sent_max_len', type=int, default=32,
                    help='max length of sentences (max source text sentence tokens)')
parser.add_argument('--dataset', type=str, default='Rest2014_hard',
                    help='dataset')

# Training
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

aspect_dict = {'Rest2014':15, 'Rest2014_hard':15, 'RestLarge':24, 'RestLarge_hard':24, 'MAMS':24}
split = {'Rest2014':3517, 'Rest2014_hard':294, 'RestLarge':4664, 'RestLarge_hard':466, 'MAMS':7089}

hps = parser.parse_args()
hps.n_aspect = aspect_dict[hps.dataset]

vocab, train, mask, lab, words = build_dataset(hps.dataset, False)
embedding_pretrained = torch.tensor(np.load('data/'+ hps.dataset + '/embedding_glove.npz')["embeddings"].astype('float32'))
embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=True)
model = HAGNN(hps, embedding)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)
criterion = torch.nn.CrossEntropyLoss()
torch.autograd.set_detect_anomaly(True)
split = split[hps.dataset] * 3



for epoch in range(0, hps.n_epochs + 1):   
    G, src, dst, label = CreateGraph(vocab, train, mask, lab, words, hps)
    
    train_src = src[:split]
    test_src = src[split:]

    train_dst = dst[:split]
    test_dst = dst[split:]

    train_label = label[:split // 3]
    test_label = label[split // 3:]
    model.train()
    
    data = model.forward(G)
    ans = data[train_src] * data[train_dst]
    ans = torch.softmax(ans, dim=1)
    ans = ans * data[train_src]
    ans = ans.sum(dim=1).squeeze()
    ans = ans.view(-1, 3)
    loss = criterion(ans, torch.tensor(train_label, dtype=torch.long))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss = float(loss.data)
    acc = (torch.softmax(ans, dim=1).argmax(dim=1) == torch.tensor(train_label)).sum().item()/len(train_label)
    print('In epoch {}, loss: {}, acc: {}'.format(epoch, train_loss, acc))
    
    model.eval()    
    with torch.no_grad():
        ans_test = data[test_src] * data[test_dst]
        ans_test = torch.softmax(ans_test, dim=1)
        ans_test = ans_test * data[test_src]
        ans_test = ans_test.sum(dim=1).squeeze()
        ans_test = ans_test.view(-1, 3)
        acc = (torch.softmax(ans_test, dim=1).argmax(dim=1) == torch.tensor(test_label)).sum().item()/len(test_label)
        macro_f1 = metrics.f1_score(test_label, torch.softmax(ans_test, dim=1).argmax(dim=1), labels=[0, 1, 2], average='macro')

        print('test acc:' + str(acc))
        print('macro-f1:' + str(macro_f1))