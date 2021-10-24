from typing import Any

import numpy as np

import torch
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import dgl

# from module.GAT import GAT, GAT_ffn
from module.GAT import WSWGAT
from module.PositionEmbedding import get_sinusoid_encoding_table
from module.Encoder import sentEncoder


class HAGNN(nn.Module):

    def __init__(self, hps, embed):
        """

        :param hps:
        :param embed: word embedding
        """
        super().__init__()
        self._hps = hps
        self._n_iter = hps.n_iter
        self._embed = embed
        self.embed_size = hps.word_emb_dim
        # sent node feature
        self._init_sn_param()
        self.n_feature_proj = nn.Linear(hps.n_feature_size * 2, hps.hidden_size, bias=False)
        self.as_feature_proj = nn.Linear(hps.n_aspect, hps.hidden_size, bias=False)
        self.word_feature_proj = nn.Linear(300, hps.hidden_size, bias=False)
        self._TFembed = nn.Linear(hps.word_emb_dim, hps.n_feature_size)
        self.fc = nn.Linear(768, 64)
        self.proj = nn.Linear(64, 8)
        # word -> sent
        embed_size = hps.word_emb_dim

        self.word2sent = WSWGAT(in_dim=embed_size,
                                out_dim=hps.hidden_size,
                                num_heads=hps.n_head,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.n_feature_size,
                                layerType="W2S"
                                )

        # sent -> word
        self.sent2word = WSWGAT(in_dim=hps.hidden_size,
                                out_dim=embed_size,
                                num_heads=6,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.n_feature_size,
                                layerType="S2W"
                                )

        # sent -> aspect
        self.sent2aspect = WSWGAT(in_dim=hps.hidden_size,
                                  out_dim=hps.hidden_size,
                                  num_heads=8,
                                  attn_drop_out=hps.atten_dropout_prob,
                                  ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                  ffn_drop_out=hps.ffn_dropout_prob,
                                  feat_embed_size=hps.n_feature_size,
                                  layerType="S2A"
                                  )

        # aspect -> sentence
        self.aspect2sent = WSWGAT(in_dim=hps.hidden_size,
                                  out_dim=hps.hidden_size,
                                  num_heads=8,
                                  attn_drop_out=hps.atten_dropout_prob,
                                  ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                  ffn_drop_out=hps.ffn_dropout_prob,
                                  feat_embed_size=hps.n_feature_size,
                                  layerType="A2S"
                                  )

        # node classification
        self.n_feature = hps.hidden_size
        self.attn_fc = nn.Linear(hps.hidden_size, 1, bias=False)

    def forward(self, graph):

        # word node init

        sent_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        word_feature = self.set_wnfeature(graph)  # [wnode, embed_size]

        # sent_feature = nn.Embedding(len(sent_id), self.n_feature).weight  # [snode, hiden_size]

        sent_feature = graph.nodes[sent_id].data["words"]
        sent_feature = self.fc(sent_feature)
        aspect_feature = self.set_asfeature(graph)

        # the start state
        word_state = word_feature
        sent_state = sent_feature
        # sent_state = self.word2sent(graph, word_feature, sent_feature)
        aspect_state = self.sent2aspect(graph, sent_state, aspect_feature)
        aspect_state = aspect_feature
        ret = self.proj(torch.cat([sent_state, aspect_state]))
        for i in range(self._n_iter):
            # sent -> word
            word_state = self.sent2word(graph, sent_state, word_state)
            # word, aspect -> sent
            sent_state = self.aspect2sent(graph, aspect_state, sent_state)
            # sent -> aspect
            aspect_state = self.sent2aspect(graph, sent_state, aspect_state)
            # sent,aspect -> edge
            ret = self.proj(torch.cat([sent_state, aspect_state]))
        return ret

    def _init_sn_param(self):
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(32, self.embed_size, padding_idx=0),
            freeze=True)
        self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
        self.lstm_hidden_state = self._hps.lstm_hidden_state
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1, batch_first=True, bidirectional=self._hps.bidirectional)
        if self._hps.bidirectional:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state * 2 * self._hps.sent_max_len, self._hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self._hps.lstm_hidden_state * self._hps.sent_max_len, self._hps.n_feature_size)
        self.ngram_enc = sentEncoder(self._hps, self._embed)

    def _sent_cnn_feature(self, graph, snode_id):
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  # [snode, embed_size]
        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        cnn_feature = self.cnn_proj(ngram_feature)
        return cnn_feature

    def _sent_lstm_feature(self, graph, snode_id):
        ngram_feature = self._embed(graph.nodes[snode_id].data["words"])  # #snode * seq_len * embed
        ngram_feature = ngram_feature.view(self._hps.sent_max_len, -1, self._hps.word_emb_dim)
        lstm_output, _ = self.lstm(ngram_feature)
        output = lstm_output.reshape(-1, self._hps.lstm_hidden_state * 2 * self._hps.sent_max_len)
        lstm_feature = self.lstm_proj(output)  # [n_nodes, n_feature_size]
        return lstm_feature

    def set_wnfeature(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)
        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
        w_embed = self._embed(wid)  # [n_wnodes, D]
        graph.nodes[wnode_id].data["embed"] = w_embed

        etf = graph.edges[wsedge_id].data["position"]
        graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(self.sent_pos_embed(etf))
        # w_embed = self.word_feature_proj(w_embed)
        return w_embed

    def set_snfeature(self, graph):
        # node feature
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        cnn_feature = self._sent_cnn_feature(graph, snode_id)
        lstm_feature = self._sent_lstm_feature(graph, snode_id)
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]
        return node_feature

    def set_asfeature(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        as_embed = graph.nodes[wnode_id].data["asembed"]
        as_embed = self.as_feature_proj(as_embed)
        return as_embed
