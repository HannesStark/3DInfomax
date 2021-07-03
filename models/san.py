import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

from commons.mol_encoder import AtomEncoder, BondEncoder
from models.base_layers import MLP

"""
    Graph Transformer Layer

"""

"""
    Util functions
"""


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}

    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}

    return func


# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """

    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}

    return func


def exp_real(field, L):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5)) / (L + 1)}

    return func


def exp_fake(field, L):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': L * torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5)) / (L + 1)}

    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = gamma
        self.full_graph = full_graph

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.E = nn.Linear(in_dim, out_dim * num_heads, bias=True)

            if self.full_graph:
                self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)

            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)

        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.E = nn.Linear(in_dim, out_dim * num_heads, bias=False)

            if self.full_graph:
                self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)

            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):

        if self.full_graph:
            real_ids = torch.nonzero(g.edata['real']).squeeze()
            fake_ids = torch.nonzero(g.edata['real'] == 0).squeeze()
        else:
            real_ids = g.edges(form='eid')

        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'), edges=real_ids)

        if self.full_graph:
            g.apply_edges(src_dot_dst('K_2h', 'Q_2h', 'score'), edges=fake_ids)

        # scale scores by sqrt(d)
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        # Use available edge features to modify the scores for edges
        g.apply_edges(imp_exp_attn('score', 'E'), edges=real_ids)

        if self.full_graph:
            g.apply_edges(imp_exp_attn('score', 'E_2'), edges=fake_ids)

        if self.full_graph:
            # softmax and scaling by gamma
            L = self.gamma
            g.apply_edges(exp_real('score', L), edges=real_ids)
            g.apply_edges(exp_fake('score', L), edges=fake_ids)

        else:
            g.apply_edges(exp('score'), edges=real_ids)

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_soft', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'z'))

    def forward(self, g, h, e):

        Q_h = self.Q(h)
        K_h = self.K(h)
        E = self.E(e)

        if self.full_graph:
            Q_2h = self.Q_2(h)
            K_2h = self.K_2(h)
            E_2 = self.E_2(e)

        V_h = self.V(h)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.edata['E'] = E.view(-1, self.num_heads, self.out_dim)

        if self.full_graph:
            g.ndata['Q_2h'] = Q_2h.view(-1, self.num_heads, self.out_dim)
            g.ndata['K_2h'] = K_2h.view(-1, self.num_heads, self.out_dim)
            g.edata['E_2'] = E_2.view(-1, self.num_heads, self.out_dim)

        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))

        return h_out


class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """

    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, dropout=0.0, layer_norm=False, batch_norm=True,
                 residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(gamma, in_dim, out_dim // num_heads, num_heads, full_graph, use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, e):
        h_in1 = h  # for first residual connection

        # multi-head attention out
        h_attn_out = self.attention(g, h, e)

        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        return h, e


class SAN(nn.Module):
    def __init__(self, GT_out_dim, readout_hidden_dim, readout_batchnorm, readout_aggregators, target_dim,
                 readout_layers, batch_norm_momentum, **kwargs):
        super().__init__()
        self.readout_aggregators = readout_aggregators
        self.gnn = SAN_NodeLPE(GT_out_dim=GT_out_dim, batch_norm_momentum=batch_norm_momentum, **kwargs)
        self.output = MLP(in_dim=GT_out_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers,
                          batch_norm_momentum=batch_norm_momentum)

    def forward(self, g):
        self.gnn(g)

        readouts_to_cat = [dgl.readout_nodes(g, 'feat', op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)


class SAN_NodeLPE(nn.Module):
    def __init__(self, node_dim, edge_dim, batch_norm_momentum, residual, in_feat_dropout, dropout,
                 layer_norm, batch_norm, gamma, full_graph, GT_hidden_dim, GT_n_heads, GT_out_dim, GT_layers,
                 LPE_n_heads, LPE_layers, LPE_dim, **kwargs):
        super().__init__()

        self.residual = residual

        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)


        self.embedding_h = AtomEncoder(emb_dim=GT_hidden_dim - LPE_dim)
        self.embedding_e_real = BondEncoder(emb_dim=GT_hidden_dim)
        self.embedding_e_fake = BondEncoder(emb_dim=GT_hidden_dim)

        self.linear_A = nn.Linear(2, LPE_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=LPE_dim, nhead=LPE_n_heads)
        self.PE_Transformer = nn.TransformerEncoder(encoder_layer, num_layers=LPE_layers)

        self.layers = nn.ModuleList([GraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph,
                                                           dropout, self.layer_norm, self.batch_norm, self.residual) for
                                     _ in range(GT_layers - 1)])

        self.layers.append(
            GraphTransformerLayer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, full_graph, dropout, self.layer_norm,
                                  self.batch_norm, self.residual))

    def forward(self, g):
        # input embedding
        h = self.embedding_h(g.ndata['feat'])
        e = self.embedding_e_real(g.edata['feat'])

        PosEnc = g.ndata['pos_enc']  # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(PosEnc)  # (Num nodes) x (Num Eigenvectors) x 2

        PosEnc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        PosEnc = torch.transpose(PosEnc, 0, 1).float()  # (Num Eigenvectors) x (Num nodes) x 2
        PosEnc = self.linear_A(PosEnc)  # (Num Eigenvectors) x (Num nodes) x PE_dim
        # 1st Transformer: Learned PE
        PosEnc = self.PE_Transformer(src=PosEnc, src_key_padding_mask=empty_mask[:, :, 0])
        # remove masked sequences
        PosEnc[torch.transpose(empty_mask, 0, 1)[:, :, 0]] = float('nan')
        # Sum pooling
        PosEnc = torch.nansum(PosEnc, 0, keepdim=False)
        # Concatenate learned PE to input embedding
        h = torch.cat((h, PosEnc), 1)

        h = self.in_feat_dropout(h)

        # Second Transformer
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['feat'] = h
