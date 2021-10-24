import dgl
import torch
import warnings
import dgl.init
import numpy as np
import torch.nn as nn
from utils import build_dataset
from pytorch_pretrained_bert.modeling import BertModel

bert = BertModel.from_pretrained('./bert_base_uncased')

def AddWordNode(vocab, G, words):
    wid2nid = {}
    nid2wid = {}
    nid = 0
    for sentid in words:
        for wid in sentid:
            if (wid not in vocab.values()) or (wid in wid2nid):
                continue
            wid2nid[wid] = nid
            nid2wid[nid] = wid
            nid += 1
    w_nodes = len(wid2nid)
    G.add_nodes(w_nodes)
    G.ndata["unit"] = torch.zeros(w_nodes)
    G.ndata["id"] = torch.LongTensor(list(nid2wid.values())) 
    G.ndata["dtype"] = torch.zeros(w_nodes)
    return wid2nid, nid2wid


def CreateGraph(vocab, input_pad, mask, aspect_list, words, hps):
    """ Create a graph for each document

    :param mask: bert mask
    :param vocab: vocabulary
    :param aspect_list: aspect label
    :param input_pad: list(list); [sentnum, wordnum]
    :return: G: dgl.DGLGraph
    """
    n_aspect = hps.n_aspect
    dataset = hps.dataset
    # embedding_pretrained = torch.tensor(np.load('data/' + dataset + '/embedding_glove.npz')["embeddings"].astype('float32'))
    # embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G = dgl.DGLGraph()
    wid2nid, nid2wid = AddWordNode(vocab, G, words)
    w_nodes = len(nid2wid)
    N = len(input_pad)
    G.add_nodes(N)
    sentid2nid = [i + w_nodes for i in range(N)]
    G.ndata["unit"][w_nodes:] = torch.ones(N)
    G.ndata["dtype"][w_nodes:] = torch.ones(N)
    input_pad = torch.tensor(input_pad)
    mask = torch.tensor(mask)
    _, pooled = bert(input_pad, attention_mask=mask, output_all_encoded_layers=False)
    for name, param in bert.named_parameters():  
        param.requires_grad = False
    # fc = nn.Linear(768, 64)
    # pooled = fc(pooled)
    G.nodes[sentid2nid].data["words"] = pooled  # [N, seq_len]
    # G.nodes[sentid2nid].data["sent_embedding"] = embedding(torch.LongTensor(input_pad))
    aspectid = [i + w_nodes + N for i in range(n_aspect)]
    G.add_nodes(n_aspect)
    G.ndata["dtype"][w_nodes+N:] = torch.ones(n_aspect) * 2
    G.nodes[aspectid].data["asembed"] = torch.zeros(n_aspect, n_aspect) + torch.eye(n_aspect)
    G.ndata["unit"][w_nodes+N:] = torch.ones(n_aspect) * 2
    src = []
    dst = []
    label = []
    for i in range(N):
        sentence = input_pad[i]
        sent_nid = sentid2nid[i]
        aspect_list_i = aspect_list[i]
        if aspect_list_i < 3:
            G.add_edges(sent_nid, aspect_list_i + w_nodes + N,
                        data={"dtype": torch.Tensor([1])})
            G.add_edges(aspect_list_i + w_nodes + N, sent_nid,
                        data={"dtype": torch.Tensor([1])})
            src.extend([i, i, i])
            dst.extend([N, N+1, N+2])
            label.extend([aspect_list_i])
        elif aspect_list_i < 6:
            G.add_edges(sent_nid, aspect_list_i + w_nodes + N,
                        data={"dtype": torch.Tensor([1])})
            G.add_edges(aspect_list_i + w_nodes + N, sent_nid,
                        data={"dtype": torch.Tensor([1])})
            src.extend([i, i, i])
            dst.extend([N+3, N+4, N+5])
            label.extend([aspect_list_i % 3])
        elif aspect_list_i < 9:
            G.add_edges(sent_nid, aspect_list_i + w_nodes + N,
                        data={"dtype": torch.Tensor([1])})
            G.add_edges(aspect_list_i + w_nodes + N, sent_nid,
                        data={"dtype": torch.Tensor([1])})
            src.extend([i, i, i])
            dst.extend([N+6, N+7, N+8])
            label.extend([aspect_list_i % 3])
        elif aspect_list_i < 12:
            G.add_edges(sent_nid, aspect_list_i + w_nodes + N,
                        data={"dtype": torch.Tensor([1])})
            G.add_edges(aspect_list_i + w_nodes + N, sent_nid,
                        data={"dtype": torch.Tensor([1])})
            src.extend([i, i, i])
            dst.extend([N+9, N+10, N+11])
            label.extend([aspect_list_i % 3])
        elif aspect_list_i < 15:
            G.add_edges(sent_nid, aspect_list_i + w_nodes + N,
                        data={"dtype": torch.Tensor([1])})
            G.add_edges(aspect_list_i + w_nodes + N, sent_nid,
                        data={"dtype": torch.Tensor([1])})
            src.extend([i, i, i])
            dst.extend([N+12, N+13, N+14])
            label.extend([aspect_list_i % 3])
        elif aspect_list_i < 18:
            G.add_edges(sent_nid, aspect_list_i + w_nodes + N,
                        data={"dtype": torch.Tensor([1])})
            G.add_edges(aspect_list_i + w_nodes + N, sent_nid,
                        data={"dtype": torch.Tensor([1])})
            src.extend([i, i, i])
            dst.extend([N+15, N+16, N+17])
            label.extend([aspect_list_i % 3])
        elif aspect_list_i < 21:
            G.add_edges(sent_nid, aspect_list_i + w_nodes + N,
                        data={"dtype": torch.Tensor([1])})
            G.add_edges(aspect_list_i + w_nodes + N, sent_nid,
                        data={"dtype": torch.Tensor([1])})
            src.extend([i, i, i])
            dst.extend([N+18, N+19, N+20])
            label.extend([aspect_list_i % 3])
        elif aspect_list_i < 24:
            G.add_edges(sent_nid, aspect_list_i + w_nodes + N,
                        data={"dtype": torch.Tensor([1])})
            G.add_edges(aspect_list_i + w_nodes + N, sent_nid,
                        data={"dtype": torch.Tensor([1])})
            src.extend([i, i, i])
            dst.extend([N+21, N+22, N+23])
            label.extend([aspect_list_i % 3])
        for index, wid in enumerate(words[i]):
            if wid in wid2nid and wid != len(vocab) - 1:
                G.add_edges(wid2nid[wid], sent_nid,
                        data={"dtype": torch.Tensor([0]), "position": torch.LongTensor([index])})
                G.add_edges(sent_nid, wid2nid[wid],
                        data={"dtype": torch.Tensor([0]), "position": torch.LongTensor([index])})

    return G, src, dst, label


if __name__ == '__main__':
    class Test():
        def __init__(self, aspects, dataset):
            self.n_aspect = aspects
            self.dataset = dataset
    vocab, train, mask, lab, words = build_dataset('Rest2014_hard', False)
    test_sample = Test(15, 'Rest2014_hard')
    g, src, dst, label = CreateGraph(vocab, train, mask, lab, words, test_sample)
    print(g)


