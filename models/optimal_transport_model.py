import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter

from commons.geomol_utils import *

from itertools import permutations
import numpy as np
import ot

from models.base_layers import MLP
from models.geomol_mpnn_ogb_feat_random_non_shared import GeomolGNNOGBFeatRandomNonShared
from models.gin_random import GNN_node_VirtualnodeRandom
from models.pna_gnn_random import PNAGNNRandom
from models.geomol_mpnn import GeomolGNN, GeomolMLP
from models.geomol_mpnn_ogb_feat import GeomolGNNOGBFeat
from models.geomol_mpnn_ogb_feat_random import GeomolGNNOGBFeatRandom
from models.pna_original_random import PNAGNNSimpleRandom
from models.gin_random import GNN_nodeRandom
from models.pna_edge_update_random import PNAGNNRandomEdgeUpdate



DEBUG_NEIGHBORHOOD_PAIRS = False


class OptimalTransportModel(nn.Module):
    def __init__(self, hyperparams,gnn_model, gnn_params, device, use_transformer = True, use_two_gnns= True, **kwargs):
        super(OptimalTransportModel, self).__init__()

        self.hidden_dim = hyperparams['hidden_dim']
        self.random_vec_dim = hyperparams['random_vec_dim']
        self.random_vec_std = hyperparams['random_vec_std']
        self.global_transformer = hyperparams['global_transformer']
        self.loss_type = hyperparams['loss_type']
        self.teacher_force = hyperparams['teacher_force']
        self.random_alpha = hyperparams['random_alpha']
        self.n_true_confs = hyperparams['n_true_confs']
        self.n_model_confs = hyperparams['n_model_confs']
        self.device = device

        self.use_two_gnns = use_two_gnns
        self.gnn = globals()[gnn_model](random_vec_dim= self.random_vec_dim, n_model_confs=self.n_model_confs, pretrain_mode=True,**gnn_params)
        if self.use_two_gnns:
            self.gnn2 = globals()[gnn_model](random_vec_dim= self.random_vec_dim, n_model_confs=self.n_model_confs, pretrain_mode=True,**gnn_params)
        self.use_gnn_output_mlp = gnn_params['hidden_dim'] != self.hidden_dim
        if self.use_gnn_output_mlp:
            self.gnn_output_mlp = MLP(in_dim=gnn_params['hidden_dim'], out_dim=self.hidden_dim, layers=1)
            if self.use_two_gnns:
                self.gnn2_output_mlp = MLP(in_dim=gnn_params['hidden_dim'], out_dim=self.hidden_dim, layers=1)

        if hyperparams['global_transformer']:
            self.global_embed = TransformerEncoderLayer(d_model=self.hidden_dim, nhead=1,
                                                        dim_feedforward=self.hidden_dim * 2,
                                                        dropout=0.0, activation='relu')
        self.use_transformer= use_transformer
        if use_transformer:
            self.encoder = TransformerEncoderLayer(d_model=self.hidden_dim * 2,
                                               nhead=hyperparams['encoder']['n_head'],
                                               dim_feedforward=self.hidden_dim * 3,
                                               dropout=0.0, activation='relu')

        self.coord_pred = GeomolMLP(in_dim=self.hidden_dim * 2, out_dim=3,
                                    num_layers=hyperparams['coord_pred']['n_layers'])
        self.d_mlp = GeomolMLP(in_dim=self.hidden_dim * 2, out_dim=1, num_layers=hyperparams['d_mlp']['n_layers'])

        self.h_mol_mlp = GeomolMLP(in_dim=self.hidden_dim, out_dim=self.hidden_dim,
                                   num_layers=hyperparams['h_mol_mlp']['n_layers'])
        if self.random_alpha:
            self.alpha_mlp = GeomolMLP(in_dim=self.hidden_dim * 3 + self.random_vec_dim, out_dim=1,
                                       num_layers=hyperparams['alpha_mlp']['n_layers'])
        else:
            self.alpha_mlp = GeomolMLP(in_dim=self.hidden_dim * 3, out_dim=1,
                                       num_layers=hyperparams['alpha_mlp']['n_layers'])
        self.c_mlp = GeomolMLP(in_dim=self.hidden_dim * 4, out_dim=1, num_layers=hyperparams['c_mlp']['n_layers'])



        self.loss = torch.nn.MSELoss(reduction='none')

        self.one_hop_loss = []
        self.two_hop_loss = []
        self.angle_loss = []
        self.dihedral_loss = []
        self.three_hop_loss = []

    def forward(self, data, dgl_graph=None, ignore_neighbors=False):
        data = data.to(self.device)
        x, edge_index, edge_attr, pos, batch, pos_mask, chiral_tag = \
            data.x, data.edge_index, data.edge_attr, data.pos, data.batch, data.pos_mask, data.chiral_tag

        if isinstance(pos, list):
            pos = torch.cat([torch.cat([p[0][i] for p in pos]).unsqueeze(1) for i in range(self.n_true_confs)], dim=1)

        # assign neighborhoods
        self.assign_neighborhoods(x, edge_index, edge_attr, batch, data)

        # calculate ground truth stats
        batched_true_stats = self.batch_true_stats(pos)
        self.true_stats = batched_true_stats

        # split into individual confs and create list of tuples
        split_true_stats = [stat.split(1, dim=-1) for stat in batched_true_stats]
        true_stats = [tuple([stat[i].squeeze(-1) for stat in split_true_stats]) for i in range(self.n_true_confs)]

        # calculate predicted model stats
        batched_model_stats = self.generate_model_prediction(x, edge_index, edge_attr, batch, chiral_tag, dgl_graph=dgl_graph)

        # split into individual confs and create list of tuples
        split_model_stats = [stat.split(1, dim=-1) for stat in batched_model_stats]
        model_stats = [tuple([stat[i].squeeze(-1) for stat in split_model_stats]) for i in range(self.n_model_confs)]
        self.model_stats = batched_model_stats

        # loss
        molecule_loss = torch.stack(
            [torch.stack([self.batch_molecule_loss(a, b, ignore_neighbors) for b in model_stats]) for a in true_stats])

        pos_mask_L2 = pos_mask.view(molecule_loss.size(2), self.n_true_confs).t()
        pos_mask_L1 = pos_mask_L2.unsqueeze(1).repeat(1, self.n_model_confs, 1)
        molecule_loss = torch.where(pos_mask_L1 == 1, molecule_loss, torch.FloatTensor([9e99]).to(self.device))

        if self.loss_type == 'implicit_mle':

            if DEBUG_NEIGHBORHOOD_PAIRS or self.teacher_force:
                L1 = torch.where(pos_mask_L2 == 1, torch.min(molecule_loss, dim=0).values,
                                 torch.FloatTensor([0]).to(self.device)).sum(dim=0) / pos_mask_L2.sum(dim=0)
            else:
                L1 = torch.min(molecule_loss, dim=0).values.sum(dim=0) / self.n_model_confs

            L2 = torch.where(pos_mask_L2 == 1, torch.min(molecule_loss, dim=1).values,
                             torch.FloatTensor([0]).to(self.device)).sum(dim=0) / pos_mask_L2.sum(dim=0)

            # logging
            self.run_writer_mle(True if L1.mean() > L2.mean() else False, molecule_loss, pos_mask_L2)
            return torch.max(L1.mean(), L2.mean())

        elif self.loss_type == 'ot_emd':

            n_true_confs_batch = data.pos_mask.view(-1, self.n_true_confs).sum(dim=1)
            H_2 = np.ones(self.n_model_confs) / self.n_model_confs
            cost_mat_detach = molecule_loss.permute(2, 0, 1).detach().cpu().numpy()
            loss = 0

            ot_mat_list = []
            for i in range(cost_mat_detach.shape[0]):

                H_1 = np.ones(n_true_confs_batch[i].item()) / n_true_confs_batch[i].item()

                if self.teacher_force:
                    cost_mat_i = cost_mat_detach[i, :n_true_confs_batch[i], :n_true_confs_batch[i]]
                    ot_mat = ot.emd(a=H_1, b=H_1, M=np.max(np.abs(cost_mat_i)) + cost_mat_i, numItermax=10000)
                    ot_mat_attached = torch.tensor(ot_mat, device=self.device, requires_grad=False).float()
                    ot_mat_list.append(ot_mat_attached)
                    loss += torch.sum(
                        ot_mat_attached * molecule_loss[:n_true_confs_batch[i], :n_true_confs_batch[i], i])
                else:
                    cost_mat_i = cost_mat_detach[i, :n_true_confs_batch[i]]
                    ot_mat = ot.emd(a=H_1, b=H_2, M=np.max(np.abs(cost_mat_i)) + cost_mat_i, numItermax=10000)
                    ot_mat_attached = torch.tensor(ot_mat, device=self.device, requires_grad=False).float()
                    ot_mat_list.append(ot_mat_attached)
                    loss += torch.sum(ot_mat_attached * molecule_loss[:n_true_confs_batch[i], :, i])

            self.run_writer_ot_emd(ot_mat_list, n_true_confs_batch)
            return loss / cost_mat_detach.shape[0]

    def assign_neighborhoods(self, x, edge_index, edge_attr, batch, data):
        """
        Initialize neighbors, dihedral pairs, masks, mapping tensors, etc.
        """
        self.neighbors = get_neighbor_ids(data)
        self.leaf_hydrogens = get_leaf_hydrogens(self.neighbors, x)
        self.dihedral_pairs = data.edge_index_dihedral_pairs

        self.n_neighborhoods = len(self.neighbors)
        self.n_dihedral_pairs = len(self.dihedral_pairs.t())

        # mask for neighbors
        self.neighbor_mask = torch.zeros([self.n_neighborhoods, 4]).to(self.device)
        # maps node index to hidden index as given by self.neighbors
        self.x_to_h_map = torch.zeros(x.size(0))

        # maps local neighborhood to batch molecule
        self.neighborhood_to_mol_map = torch.zeros(self.n_neighborhoods, dtype=torch.int64).to(self.device)

        for i, (a, n) in enumerate(self.neighbors.items()):
            self.x_to_h_map[a] = i
            self.neighbor_mask[i, 0:len(n)] = 1
            self.leaf_hydrogens[a] = self.leaf_hydrogens[a] * True if self.leaf_hydrogens[a].sum() > 1 else \
                self.leaf_hydrogens[a] * False
            self.neighborhood_to_mol_map[i] = batch[a]

        # maps which atom in (x,y) corresponds to the same atom in (y,x) for each dihedral pair
        self.x_map_to_neighbor_y = torch.zeros([self.n_dihedral_pairs, 4]).to(self.device)
        self.y_map_to_neighbor_x = torch.zeros([self.n_dihedral_pairs, 4]).to(self.device)

        # neighbor mask but for dihedral pairs
        self.dihedral_x_mask = torch.zeros([self.n_dihedral_pairs, 4]).to(self.device)
        self.dihedral_y_mask = torch.zeros([self.n_dihedral_pairs, 4]).to(self.device)

        # maps neighborhood pair to batch molecule
        self.neighborhood_pairs_to_mol_map = torch.zeros(self.n_dihedral_pairs, dtype=torch.int64).to(self.device)
        for i, (s, e) in enumerate(self.dihedral_pairs.t()):
            # this indicates which neighbor is the correct x <--> y map (see overleaf doc)
            self.x_map_to_neighbor_y[i, 0:len(self.neighbors[s.item()])] = self.neighbors[s.item()] == e
            self.y_map_to_neighbor_x[i, 0:len(self.neighbors[e.item()])] = self.neighbors[e.item()] == s

            # create dihedral masks
            self.dihedral_x_mask[i, :] = self.neighbor_mask[self.x_to_h_map[s].long()]
            self.dihedral_y_mask[i, :] = self.neighbor_mask[self.x_to_h_map[e].long()]

            self.neighborhood_pairs_to_mol_map[i] = batch[s]

        # calculate dihedral mask
        self.dihedral_mask = torch.bmm(self.dihedral_x_mask[~self.x_map_to_neighbor_y.bool()].view(-1, 3, 1),
                                       self.dihedral_y_mask[~self.y_map_to_neighbor_x.bool()].view(-1, 1, 3)).view(-1,
                                                                                                                   9)

    def embed(self, x, edge_index, edge_attr, batch, dgl_graph):

        rand_dist = torch.distributions.normal.Normal(loc=0, scale=self.random_vec_std)
        # rand_dist = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        rand_x = rand_dist.sample([x.size(0), self.n_model_confs, self.random_vec_dim]).squeeze(-1).to(
            self.device)  # added squeeze
        rand_edge = rand_dist.sample([edge_attr.size(0), self.n_model_confs, self.random_vec_dim]).squeeze(-1).to(
            self.device)  # added squeeze

        if dgl_graph != None:
            dgl_graph_clone = dgl_graph.clone()
        else:
            dgl_graph_clone = None
        # gnn
        x1, _ = self.gnn(x=x, edge_index=edge_index, edge_attr=edge_attr, rand_x=rand_x, rand_edge=rand_edge, dgl_graph=dgl_graph)
        if self.use_two_gnns:
            x2, _ = self.gnn2(x=x, edge_index=edge_index, edge_attr=edge_attr, rand_x=rand_x, rand_edge=rand_edge, dgl_graph=dgl_graph_clone)
        else:
            x2 = x1.clone()

        if self.use_gnn_output_mlp:
            x1 = self.gnn_output_mlp(x1)
            if self.use_two_gnns:
                x2 = self.gnn2_output_mlp(x2)
            else:
                x2 = x1.clone()

        if self.global_transformer:

            # global embeddings with transformer
            n_max = batch.bincount().max()
            x_transformer, x_mask = tg.utils.to_dense_batch(x2, batch)

            x_transformer = x_transformer.permute(1, 0, 2, 3).reshape(n_max, -1, self.hidden_dim)
            x_transformer_mask = x_mask.unsqueeze(1).repeat(1, self.n_model_confs, 1).view(-1, n_max)

            x_global = self.global_embed(x_transformer, src_key_padding_mask=~x_transformer_mask).view(
                n_max, max(batch) + 1, self.n_model_confs, -1).permute(1, 0, 2, 3) * \
                       x_transformer_mask.view(max(batch) + 1, n_max, self.n_model_confs, 1)

            # global reps for torsions
            h_mol = self.h_mol_mlp(x_global.sum(dim=1))

            # reshape to sparse
            x2 = x_global[x_mask, :]

        else:
            h_mol = self.h_mol_mlp(global_add_pool(x2, batch))

        return x1, x2, h_mol

    def model_local_stats(self, x, chiral_tag):

        n_h = torch.zeros([self.n_neighborhoods, 4, self.n_model_confs, self.hidden_dim]).to(self.device)
        x_h = torch.zeros([self.n_neighborhoods, self.n_model_confs, self.hidden_dim]).to(self.device)

        for i, (a, n) in enumerate(self.neighbors.items()):
            n_h[i, 0:len(n), :] = x[n]
            x_h[i, :] = x[a]

        # unit direction predictions (make sure to use opposite mask bc pytorch transformer is dumb)
        h = torch.cat([n_h, x_h.unsqueeze(1).repeat(1, 4, 1, 1)], dim=-1) * self.neighbor_mask.unsqueeze(-1).unsqueeze(
            -1)

        # prepare inputs for transformer
        h_ = h.permute(1, 0, 2, 3).reshape(4, self.n_neighborhoods * self.n_model_confs,
                                           self.hidden_dim * 2)  # CHECK RESHAPE OP
        h_mask = self.neighbor_mask.bool().unsqueeze(1).repeat(1, self.n_model_confs, 1).view(
            self.n_neighborhoods * self.n_model_confs, 4)
        if self.use_transformer:
            h_new = self.encoder(h_, src_key_padding_mask=~h_mask).view(4, self.n_neighborhoods, self.n_model_confs,
                                                                    self.hidden_dim * 2).permute(1, 0, 2, 3) \
                * self.neighbor_mask.unsqueeze(-1).unsqueeze(-1)
        else:
            h_new = h_
        unit_normals = self.coord_pred(h_new) * self.neighbor_mask.unsqueeze(-1).unsqueeze(-1)

        # tetrahedral chiral corrections
        # if chirality = CW and SV = +1  -> flip sign (z = -1)
        # if chirality = CW and SV = -1  -> keep sign (z = +1)
        # if chirality = CCW and SV = +1 -> keep sign (z = +1)
        # if chirality = CCW and SV = -1 -> flip sign (z = -1)
        # can accomplish above if CW = -1 and CCW = +1 -> z = chirality * SV
        chiral_tag_neighborhoods = chiral_tag[list(self.neighbors.keys())]
        chiral_ids = torch.nonzero(chiral_tag_neighborhoods).squeeze(-1)  # potential bugs if no chiral centers?
        if len(chiral_ids) != 0:
            signed_vols = signed_volume(unit_normals[chiral_ids])
            chiral_tag_confs = chiral_tag_neighborhoods[chiral_ids].unsqueeze(-1).repeat(1, self.n_model_confs)
            z_flip = signed_vols * chiral_tag_confs
            flip_mat = torch.diag_embed(
                torch.stack([torch.ones_like(z_flip), torch.ones_like(z_flip), z_flip]).permute(1, 2, 0)).unsqueeze(1)
            unit_normals[chiral_ids] = torch.matmul(flip_mat, unit_normals[chiral_ids].unsqueeze(-1)).squeeze(-1)

        # distance predictions
        h_flipped = torch.cat([x_h.unsqueeze(1).repeat(1, 4, 1, 1), n_h], dim=-1) * self.neighbor_mask.unsqueeze(
            -1).unsqueeze(-1)
        d_preds = F.softplus(self.d_mlp(h) + self.d_mlp(h_flipped)) * self.neighbor_mask.unsqueeze(-1).unsqueeze(-1)

        # coordinate calculation
        if DEBUG_NEIGHBORHOOD_PAIRS:
            self.model_local_coords = self.true_local_coords[:, 0]
        else:
            self.model_local_coords = unit_normals / (
                    torch.linalg.norm(unit_normals, dim=-1, keepdim=True) + 1e-10) * d_preds

        # calculate local stats
        model_one_hop, model_two_hop, model_angles = batch_local_stats_from_coords(self.model_local_coords,
                                                                                   self.neighbor_mask)

        if self.teacher_force:
            R = random_rotation_matrix([self.n_neighborhoods, 1, self.n_model_confs]).to(self.device)
            self.model_local_coords = torch.matmul(R, self.true_local_coords[:, 0].unsqueeze(-1)).squeeze(-1)

        return model_one_hop, model_two_hop, model_angles

    def ground_truth_local_stats(self, pos):
        """
        Compute true one-hop, two-hop, and angle local stats. Note that the second dimension of the local coordinates
        is 6 to account for possible symmetric hydrogens. The max number of symmetric leaf hydrogens is 3, which leads
        to a max of 6 permutations (our model doesn't work for methane). This dimension captures these symmetric
        hydrogen permutations.

        :param pos: coordinates (n_atoms, n_true_confs, 3)
        :return: tuple of true stats (one-hop, two-hop, and angles)
            true_one_hop (n_neighborhoods, 6, 4, n_true_confs)
            true_two_hop (n_neighborhoods, 6, 4, 4, n_true_confs)
            true_angles (n_neighborhoods, 6, 6, n_true_confs)
        """

        n_neighborhoods = len(self.neighbors)
        self.true_local_coords = torch.zeros(n_neighborhoods, 6, 4, self.n_true_confs, 3).to(self.device)

        for i, (a, n) in enumerate(self.neighbors.items()):

            # permutations for symmetric hydrogens
            n_perms = n.unsqueeze(0).repeat(6, 1)
            perms = torch.tensor(list(permutations(n[self.leaf_hydrogens[a]]))).to(self.device)
            if perms.size(1) != 0:
                n_perms[0:len(perms), self.leaf_hydrogens[a]] = perms

            # keep it local
            self.true_local_coords[i, :, 0:len(n)] = pos[n_perms] - pos[a]

        # calculate true local stats
        true_one_hop, true_two_hop, true_angles = batch_local_stats_from_coords(self.true_local_coords,
                                                                                self.neighbor_mask)

        return true_one_hop, true_two_hop, true_angles

    def local_loss(self, true_one_hop, true_two_hop, true_angles, model_one_hop, model_two_hop, model_angles):

        # bond distance loss
        model_one_hop_perms = model_one_hop.unsqueeze(1).repeat(1, 6, 1)
        one_hop_loss_perm = self.loss(true_one_hop, model_one_hop_perms).sum(dim=-1) / (
                true_one_hop.bool().sum(dim=-1) + 1e-10)
        one_hop_loss = scatter(one_hop_loss_perm.min(dim=-1).values, self.neighborhood_to_mol_map, reduce="mean")

        # two-hop distance loss
        model_two_hop_perms = model_two_hop.unsqueeze(1).repeat(1, 6, 1, 1)
        two_hop_loss_perm = self.loss(true_two_hop, model_two_hop_perms).sum(dim=[-1, -2]) / (
                torch.sum(true_two_hop > 1e-8, dim=[-1, -2]) + 1e-10)
        two_hop_loss = scatter(two_hop_loss_perm.min(dim=-1).values, self.neighborhood_to_mol_map, reduce="mean")

        # bending angles loss
        model_angles_perms = model_angles.unsqueeze(1).repeat(1, 6, 1)
        angle_loss_perm = torch.sum(von_Mises_loss(true_angles, model_angles_perms) * true_angles.bool(), dim=-1) / (
                true_angles.bool().sum(dim=-1) + 1e-10)
        angle_loss = scatter(angle_loss_perm.max(dim=-1).values, self.neighborhood_to_mol_map, reduce="mean")

        return one_hop_loss, two_hop_loss, angle_loss

    def model_pair_stats(self, x, batch, h_mol):
        """
        Compute dihedral angles and three-hop distances for model conformers. Each stat has size 9 for the second
        dimension since there are 9 possible permuations between sets of neighbors (X and Y have max 3 neighbors each).

        :param x: atom representations (n_atoms, n_model_confs, hidden_dim/2)
        :param batch: mapping of atom to molecule (n_atoms)
        :param h_mol: molecule representations (n_batch, n_model_confs, hidden_dim/2)
        :return: tuple of true stats (dihedral and three-hop), each with size (n_dihedral_pairs, 9, n_true_confs)
        """

        dihedral_x_neighbors = torch.zeros([self.n_dihedral_pairs, 4, self.n_model_confs, 3]).to(self.device)
        dihedral_x_node_reps = torch.zeros([self.n_dihedral_pairs, self.n_model_confs, self.hidden_dim]).to(self.device)
        dihedral_x_neighbor_reps = torch.zeros([self.n_dihedral_pairs, 4, self.n_model_confs, self.hidden_dim]).to(
            self.device)

        dihedral_y_neighbors = torch.zeros([self.n_dihedral_pairs, 4, self.n_model_confs, 3]).to(self.device)
        dihedral_y_node_reps = torch.zeros([self.n_dihedral_pairs, self.n_model_confs, self.hidden_dim]).to(self.device)
        dihedral_y_neighbor_reps = torch.zeros([self.n_dihedral_pairs, 4, self.n_model_confs, self.hidden_dim]).to(
            self.device)

        for i, (s, e) in enumerate(self.dihedral_pairs.t()):
            # get dihedral node embedded representations from gnn
            dihedral_x_node_reps[i, :] = x[s]
            dihedral_y_node_reps[i, :] = x[e]

            # get dihedral node neighbor predicted coordinates
            dihedral_x_neighbors[i, :, :] = self.model_local_coords[self.x_to_h_map[s].long()]
            dihedral_y_neighbors[i, :, :] = self.model_local_coords[self.x_to_h_map[e].long()]

            # get dihedral neighbor embeddings
            x_n_ids = self.neighbors[s.item()]
            y_n_ids = self.neighbors[e.item()]
            dihedral_x_neighbor_reps[i, :len(x_n_ids)] = x[x_n_ids]
            dihedral_y_neighbor_reps[i, :len(y_n_ids)] = x[y_n_ids]

        # align neighbor coords
        dihedral_node_reps = dihedral_x_node_reps, dihedral_y_node_reps
        dihedral_neighbors = dihedral_x_neighbors, dihedral_y_neighbors
        dihedral_neighbor_reps = dihedral_x_neighbor_reps, dihedral_y_neighbor_reps
        q_Z_prime, p_T_alpha, p_Y_alpha, q_Z_translated = self.align_dihedral_neighbors(dihedral_node_reps,
                                                                                        dihedral_neighbors,
                                                                                        batch,
                                                                                        h_mol,
                                                                                        dihedral_neighbor_reps)

        # calculate model dihedrals
        pT_idx, qZ_idx = torch.cartesian_prod(torch.arange(3), torch.arange(3)).chunk(2, dim=-1)
        pT_idx = pT_idx.squeeze(-1)
        qZ_idx = qZ_idx.squeeze(-1)

        p_T_alpha_combos = p_T_alpha[:, pT_idx, :]
        q_Z_translated_combos = q_Z_translated[:, qZ_idx, :]
        p_Y_alpha_combos = p_Y_alpha.unsqueeze(1).repeat(1, 9, 1, 1)

        model_dihedrals_sin, model_dihedrals_cos = batch_dihedrals(p_T_alpha_combos, torch.zeros_like(p_Y_alpha_combos),
                                                                   p_Y_alpha_combos, q_Z_translated_combos)
        model_dihedrals_sin = model_dihedrals_sin * self.dihedral_mask.unsqueeze(-1)
        model_dihedrals_cos = model_dihedrals_cos * self.dihedral_mask.unsqueeze(-1)
        model_dihedrals = torch.stack([model_dihedrals_sin, model_dihedrals_cos], dim=0)
        # model_dihedrals = batch_vector_angles(p_T_alpha_combos, torch.zeros_like(p_T_alpha_combos), p_Y_alpha_combos, q_Z_translated_combos).view(-1, 9, self.n_model_confs) * self.dihedral_mask.unsqueeze(-1)

        # three-hop distances
        model_three_hop = torch.linalg.norm(p_T_alpha_combos - q_Z_translated_combos,
                                            dim=-1) * self.dihedral_mask.unsqueeze(-1)

        # for inference
        self.p_coords = dihedral_x_neighbors  # this changed from (batch, 4, 3) -> (batch, 4, n_model_confs, 3)
        self.q_coords = dihedral_y_neighbors
        self.qZ_final = q_Z_translated
        self.pT_final = p_T_alpha
        self.pY_final = p_Y_alpha

        return model_dihedrals, model_three_hop

    def ground_truth_pair_stats(self, pos):
        """
        Compute dihedral angles and three-hop distances for ground truth conformers. Each stat has size 9 for the second
        dimension since there are 9 possible permuations between sets of neighbors (X and Y have max 3 neighbors each).
        Each stat has 6 for the third dimension because of potential permutations with symmetric hydrogens

        :param pos: coordinates (n_atoms, n_true_confs, 3)
        :return: tuple of true stats (dihedral and three-hop), each with size (n_dihedral_pairs, 9, 6, n_true_confs);
            dihedrals have an extra dimension of 2 at the beginning indicating sin/cos of angle
        """

        n_dihedral_pairs = len(self.dihedral_pairs.t())
        true_dihedral_coords = torch.zeros([n_dihedral_pairs, 4, 4, 6, self.n_true_confs, 3]).to(self.device)

        for i, (s, e) in enumerate(self.dihedral_pairs.t()):
            # construct true coordinates (order is x_n, x, y, y_n)
            x_neighbor_map_perms = self.neighbors[s.item()].unsqueeze(1).repeat(1, 6)
            y_neighbor_map_perms = self.neighbors[e.item()].unsqueeze(1).repeat(1, 6)

            # permutations for symmetric hydrogens
            x_perms = torch.tensor(list(permutations(self.neighbors[s.item()][self.leaf_hydrogens[s.item()]]))).t().to(
                self.device)
            y_perms = torch.tensor(list(permutations(self.neighbors[e.item()][self.leaf_hydrogens[e.item()]]))).t().to(
                self.device)

            if x_perms.size(0) != 0:
                x_neighbor_map_perms[self.leaf_hydrogens[s.item()], 0:x_perms.size(1)] = x_perms
            if y_perms.size(0) != 0:
                y_neighbor_map_perms[self.leaf_hydrogens[e.item()], 0:y_perms.size(1)] = y_perms

            true_dihedral_coords[i, 0, 0:x_neighbor_map_perms.size(0)] = pos[x_neighbor_map_perms]
            true_dihedral_coords[i, 1] = pos[s].unsqueeze(0).unsqueeze(0).repeat(4, 6, 1, 1)
            true_dihedral_coords[i, 2] = pos[e].unsqueeze(0).unsqueeze(0).repeat(4, 6, 1, 1)
            true_dihedral_coords[i, 3, 0:y_neighbor_map_perms.size(0)] = pos[y_neighbor_map_perms]

        # get true dihedral coords
        pT_idx, qZ_idx = torch.cartesian_prod(torch.arange(3), torch.arange(3)).chunk(2, dim=-1)
        pT_idx = pT_idx.squeeze(-1)
        qZ_idx = qZ_idx.squeeze(-1)

        true_dihedral_xn_coords = true_dihedral_coords[:, 0][~self.x_map_to_neighbor_y.bool(), :].view(-1, 3, 6,
                                                                                                       self.n_true_confs,
                                                                                                       3)[:, pT_idx, :]
        true_dihedral_x_coords = true_dihedral_coords[:, 1, 0:1].repeat(1, 9, 1, 1, 1)
        true_dihedral_y_coords = true_dihedral_coords[:, 2, 0:1].repeat(1, 9, 1, 1, 1)
        true_dihedral_yn_coords = true_dihedral_coords[:, 3][~self.y_map_to_neighbor_x.bool(), :].view(-1, 3, 6,
                                                                                                       self.n_true_confs,
                                                                                                       3)[:, qZ_idx, :]

        # calculate true dihedrals
        true_dihedrals_sin, true_dihedrals_cos = batch_dihedrals(true_dihedral_xn_coords, true_dihedral_x_coords,
                                                                 true_dihedral_y_coords, true_dihedral_yn_coords)
        true_dihedrals_sin = true_dihedrals_sin * self.dihedral_mask.unsqueeze(-1).unsqueeze(-1)
        true_dihedrals_cos = true_dihedrals_cos * self.dihedral_mask.unsqueeze(-1).unsqueeze(-1)
        true_dihedrals = torch.stack([true_dihedrals_sin, true_dihedrals_cos], dim=0)
        # true_dihedrals = batch_vector_angles(true_dihedral_xn_coords, true_dihedral_x_coords, true_dihedral_y_coords,
        #                                      true_dihedral_yn_coords).view(-1, 9, 6, self.n_true_confs) * self.dihedral_mask.unsqueeze(-1).unsqueeze(-1)

        # calculate true three-hop distances
        true_three_hop = torch.linalg.norm(true_dihedral_xn_coords - true_dihedral_yn_coords,
                                           dim=-1) * self.dihedral_mask.unsqueeze(-1).unsqueeze(-1)

        # note that these are NOT translated!
        self.true_p_coords = true_dihedral_xn_coords
        self.true_x_coords = true_dihedral_x_coords
        self.true_y_coords = true_dihedral_y_coords
        self.true_q_coords = true_dihedral_yn_coords

        return true_dihedrals, true_three_hop

    def pair_loss(self, true_dihedrals, model_dihedrals, true_three_hop, model_three_hop):
        """
        Compute the loss between masked model and masked ground truth dihedrals and three-hop distances. Each stat has
        size 9 for the second dimension since there are 9 possible permuations between sets of neighbors (X and Y have
        max 3 neighbors each), and the true stats have 6 for the third dimension because of possible permutations with
        symmetric hydrogens

        :param true_dihedrals: (2, n_dihedral_pairs, 9, 6)
        :param model_dihedrals: (2, n_dihedral_pairs, 9)
        :param true_three_hop: (n_dihedral_pairs, 9, 6)
        :param model_three_hop: (n_dihedral_pairs, 9)
        :return: tuple of molecule losses (dihedral and three-hop), each with size (n_batch)
        """

        # dihedral loss
        model_dihedrals_perms = model_dihedrals.unsqueeze(-1).repeat(1, 1, 1, 6)
        dihedral_loss_perms = torch.sum(von_Mises_loss(true_dihedrals[1], model_dihedrals_perms[1], true_dihedrals[0],
                                                       model_dihedrals_perms[0]) * self.dihedral_mask.unsqueeze(-1),
                                        dim=-2) / (self.dihedral_mask.sum(dim=-1, keepdim=True) + 1e-10)
        dihedral_loss = scatter(dihedral_loss_perms.max(dim=-1).values, self.neighborhood_pairs_to_mol_map,
                                reduce="mean")

        # three-hop distance loss
        model_three_hop_perms = model_three_hop.unsqueeze(-1).repeat(1, 1, 6)
        three_hop_loss_perms = self.loss(true_three_hop, model_three_hop_perms).sum(dim=-2) / (
                self.dihedral_mask.sum(dim=-1, keepdim=True) + 1e-10)
        three_hop_loss = scatter(three_hop_loss_perms.min(dim=-1).values, self.neighborhood_pairs_to_mol_map,
                                 reduce="mean")

        return dihedral_loss, three_hop_loss

    def align_dihedral_neighbors(self, dihedral_node_reps, dihedral_neighbors, batch, h_mol, dihedral_neighbor_reps):
        """
        Performs the alignment procedure between dihedral pairs by first rotating X and Y by predicted H_x and H_y,
        respectively, rotating X by H_alpha, and finally flipping and translating Y along the x-axis

        :param dihedral_node_reps: tuple of embedded X and Y atom representations, each of size
            (n_dihedral_pairs, n_model_confs, hidden_dim/2)
        :param dihedral_neighbors: tuple of predicted neighbor local coordinates for X and Y, each of size
            (n_dihedral_pairs, 4, n_model_confs, 3)
        :param batch: mapping of atom to molecule (n_atoms)
        :param h_mol: embedded molecule representations (n_batch, n_model_confs, hidden_dim/2)
        :return: tuple of aligned coordinates
            q_Z_prime (n_dihedral_pairs, 3, n_model_confs, 3)
            p_T_alpha (n_dihedral_pairs, 3, n_model_confs, 3)
            p_Y_alpha (n_dihedral_pairs, n_model_confs, 3)
            q_Z_translated (n_dihedral_pairs, 3, n_model_confs, 3)
        """

        # unpack
        dihedral_x_node_reps, dihedral_y_node_reps = dihedral_node_reps  # (n_dihedral_pairs, n_model_confs, hidden_dim/2)
        dihedral_x_neighbors, dihedral_y_neighbors = dihedral_neighbors  # (n_dihedral_pairs, 4, n_model_confs, 3)
        dihedral_x_neighbor_reps, dihedral_y_neighbor_reps = dihedral_neighbor_reps  # (n_dihedral_pairs, 4, n_model_confs, hidden_dim/2)

        # calculate rotation matrix
        Hx = rotation_matrix_v2(dihedral_x_neighbors, self.dihedral_x_mask, self.x_map_to_neighbor_y)
        Hy = rotation_matrix_v2(dihedral_y_neighbors, self.dihedral_y_mask, self.y_map_to_neighbor_x)
        # (n_dihedral_pairs, n_model_confs, 3, 3)

        # rotate
        p_H = torch.matmul(Hx.unsqueeze(1), dihedral_x_neighbors.unsqueeze(-1)).squeeze(-1)
        q_H = torch.matmul(Hy.unsqueeze(1), dihedral_y_neighbors.unsqueeze(-1)).squeeze(-1)
        # (n_dihedral_pairs, 4, n_model_confs, 3)

        # extract nodes and neighbors
        p_T_prime = p_H[~self.x_map_to_neighbor_y.bool()].view(-1, 3, self.n_model_confs, 3)
        q_Z_prime = q_H[~self.y_map_to_neighbor_x.bool()].view(-1, 3, self.n_model_confs, 3)

        p_Y_prime = p_H[self.x_map_to_neighbor_y.bool()]
        q_X_prime = q_H[self.y_map_to_neighbor_x.bool()]

        transform_matrix = torch.diag(torch.tensor([-1., -1., 1.]).to(self.device)).unsqueeze(0).unsqueeze(0).unsqueeze(
            0)
        q_Z_translated = torch.matmul(transform_matrix, q_Z_prime.unsqueeze(-1)).squeeze(-1) + p_Y_prime.unsqueeze(
            1)  # broadcast over not coordinates

        # calculate alpha
        dihedral_h_mol = h_mol[batch[self.dihedral_pairs[0]]]  # (n_dihedral_pairs, n_model_confs. hidden_dim/2)

        # more stochasticity!
        if self.random_alpha:
            rand_dist = torch.distributions.normal.Normal(loc=0, scale=self.random_vec_std)
            rand_alpha = rand_dist.sample([self.n_dihedral_pairs, self.n_model_confs, self.random_vec_dim]).squeeze(
                -1).to(self.device)
            alpha = self.alpha_mlp(
                torch.cat([dihedral_x_node_reps, dihedral_y_node_reps, dihedral_h_mol, rand_alpha], dim=-1)) + \
                    self.alpha_mlp(
                        torch.cat([dihedral_y_node_reps, dihedral_x_node_reps, dihedral_h_mol, rand_alpha], dim=-1))
        else:
            alpha = self.alpha_mlp(torch.cat([dihedral_x_node_reps, dihedral_y_node_reps, dihedral_h_mol], dim=-1)) + \
                    self.alpha_mlp(torch.cat([dihedral_y_node_reps, dihedral_x_node_reps, dihedral_h_mol], dim=-1))
        alpha = alpha.view(self.n_dihedral_pairs, self.n_model_confs, 1)
        self.v_star = torch.cat([torch.cos(alpha), torch.sin(alpha)], dim=-1)

        # calculate current dihedral
        pT_idx, qZ_idx = torch.cartesian_prod(torch.arange(3), torch.arange(3)).chunk(2, dim=-1)
        pT_idx = pT_idx.squeeze(-1)
        qZ_idx = qZ_idx.squeeze(-1)
        XYTi_XYZj_curr_sin, XYTi_XYZj_curr_cos = batch_dihedrals(p_T_prime[:, pT_idx],
                                                                 torch.zeros_like(p_Y_prime).unsqueeze(1).repeat(1, 9,
                                                                                                                 1, 1),
                                                                 p_Y_prime.unsqueeze(1).repeat(1, 9, 1, 1),
                                                                 q_Z_translated[:, qZ_idx])

        # get c coefficients
        p_reps = dihedral_x_neighbor_reps[~self.x_map_to_neighbor_y.bool()].view(-1, 3, self.n_model_confs,
                                                                                 self.hidden_dim)
        q_reps = dihedral_y_neighbor_reps[~self.y_map_to_neighbor_x.bool()].view(-1, 3, self.n_model_confs,
                                                                                 self.hidden_dim)
        cx_reps = dihedral_x_node_reps.unsqueeze(1).repeat(1, 9, 1, 1)
        cy_reps = dihedral_y_node_reps.unsqueeze(1).repeat(1, 9, 1, 1)
        self.c_ij = self.c_mlp(torch.cat([p_reps[:, pT_idx], cx_reps, q_reps[:, qZ_idx], cy_reps], dim=-1)) + \
                    self.c_mlp(torch.cat([q_reps[:, qZ_idx], cy_reps, p_reps[:, pT_idx], cx_reps], dim=-1))

        # calculate gamma sin and cos
        A_ij = self.build_A_matrix(XYTi_XYZj_curr_sin, XYTi_XYZj_curr_cos) * self.dihedral_mask.unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1)
        A_curr = torch.sum(A_ij * self.c_ij.unsqueeze(-1), dim=1)
        determinants = torch.det(A_curr) + 1e-10
        A_curr_inv_ = A_curr.view(self.n_dihedral_pairs, self.n_model_confs, 4)[:, :, [3, 1, 2, 0]] * torch.tensor(
            [[[1., -1., -1., 1.]]]).to(self.device)
        A_curr_inv = (A_curr_inv_ / determinants.unsqueeze(-1)).view(self.n_dihedral_pairs, self.n_model_confs, 2, 2)

        A_curr_inv_v_star = torch.matmul(A_curr_inv, self.v_star.unsqueeze(-1)).squeeze(-1)
        v_gamma = A_curr_inv_v_star / (A_curr_inv_v_star.norm(dim=-1, keepdim=True) + 1e-10)
        gamma_cos, gamma_sin = v_gamma.split(1, dim=-1)

        # rotate p_coords by gamma
        H_gamma = self.build_alpha_rotation(gamma_sin.squeeze(-1), gamma_cos.squeeze(-1))
        p_T_alpha = torch.matmul(H_gamma.unsqueeze(1), p_T_prime.unsqueeze(-1)).squeeze(-1)

        return q_Z_prime, p_T_alpha, p_Y_prime, q_Z_translated

    def build_alpha_rotation(self, alpha, alpha_cos=None):
        """
        Builds the alpha rotation matrix

        :param alpha: predicted values of torsion parameter alpha (n_dihedral_pairs, n_model_confs)
        :return: alpha rotation matrix (n_dihedral_pairs, n_model_confs, 3, 3)
        """
        H_alpha = torch.FloatTensor([[[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]]).repeat(self.n_dihedral_pairs,
                                                                                  self.n_model_confs, 1, 1).to(
            self.device)

        if torch.is_tensor(alpha_cos):
            H_alpha[:, :, 1, 1] = alpha_cos
            H_alpha[:, :, 1, 2] = -alpha
            H_alpha[:, :, 2, 1] = alpha
            H_alpha[:, :, 2, 2] = alpha_cos
        else:
            H_alpha[:, :, 1, 1] = torch.cos(alpha)
            H_alpha[:, :, 1, 2] = -torch.sin(alpha)
            H_alpha[:, :, 2, 1] = torch.sin(alpha)
            H_alpha[:, :, 2, 2] = torch.cos(alpha)

        return H_alpha

    def build_A_matrix(self, curr_sin, curr_cos):

        A_ij = torch.FloatTensor([[[[[0, 0], [0, 0]]]]]).repeat(self.n_dihedral_pairs, 9, self.n_model_confs, 1, 1).to(
            self.device)
        A_ij[:, :, :, 0, 0] = curr_cos
        A_ij[:, :, :, 0, 1] = curr_sin
        A_ij[:, :, :, 1, 0] = curr_sin
        A_ij[:, :, :, 1, 1] = -curr_cos

        return A_ij

    def batch_model_stats(self, x1, x2, batch, h_mol, chiral_tag):
        """
        Converts input atom and molecular representations to model predictions of distances, angles, and torsions

        :param x: atom representations (n_atoms, n_model_confs, hidden_dim/2)
        :param batch: mapping of atom to molecule (n_atoms)
        :param h_mol: molecule representations (n_batch, n_model_confs, hidden_dim/2)
        :return:
        """

        model_one_hop, model_two_hop, model_angles = self.model_local_stats(x1, chiral_tag)
        model_dihedrals, model_three_hop = self.model_pair_stats(x2, batch, h_mol)

        return model_one_hop, model_two_hop, model_angles, model_dihedrals, model_three_hop

    def batch_true_stats(self, pos):
        """
        Converts input coordinates to true distances, angles, and torsions

        :param pos: true conformer coordinates (n_atoms, n_true_confs, 3)
        :return: tuple of true stat tensors (len 5)
        """

        true_one_hop, true_two_hop, true_angles = self.ground_truth_local_stats(pos)
        true_dihedrals, true_three_hop = self.ground_truth_pair_stats(pos)

        return true_one_hop, true_two_hop, true_angles, true_dihedrals, true_three_hop

    def generate_model_prediction(self, x, edge_index, edge_attr, batch, chiral_tag, dgl_graph):
        """
        Run one forward pass of the model to predict stats

        :param x: atom representations (n_atoms, n_model_confs, hidden_dim/2)
        :param edge_index: directed mapping of atom indices to each other to indicate bonds (2, n_bonds)
        :param edge_attr: bond representations (n_bonds, n_model_confs, hidden_dim/2)
        :param batch: mapping of atom to molecule (n_atoms)
        :return: tuple of model stat tensors (len 5)
        """

        # embed inputs
        x1, x2, h_mol = self.embed(x, edge_index, edge_attr, batch, dgl_graph=dgl_graph)

        # calculate stats (distance, angles, torsions)
        stats = self.batch_model_stats(x1, x2, batch, h_mol, chiral_tag)

        return stats

    def batch_molecule_loss(self, true_stats, model_stats, ignore_neighbors):
        """
        Compute loss for one pair of model/true molecules

        :param true_stats: tuple of masked true stat tensors (len 5)
        :param model_stats: tuple of masked model stat tensors (len 5)
            one-hop: (n_neighborhoods, 4)
            two-hop: (n_neighborhoods, 4, 4)
            angle: (n_neighborhoods, 6)
            dihedral: (2, n_dihedral_pairs, 9)
            three-hop: (n_dihedral_pairs, 9)
        :return: molecular loss for the batch (n_batch)
        """

        # unpack stats
        model_one_hop, model_two_hop, model_angles, model_dihedrals, model_three_hop = model_stats
        true_one_hop, true_two_hop, true_angles, true_dihedrals, true_three_hop = true_stats

        # calculate losses
        one_hop_loss, two_hop_loss, angle_loss = self.local_loss(true_one_hop, true_two_hop, true_angles,
                                                                 model_one_hop, model_two_hop, model_angles)
        dihedral_loss, three_hop_loss = self.pair_loss(true_dihedrals, model_dihedrals, true_three_hop, model_three_hop)

        # writing
        self.one_hop_loss.append(one_hop_loss)
        self.two_hop_loss.append(two_hop_loss)
        self.angle_loss.append(angle_loss)
        self.dihedral_loss.append(dihedral_loss)
        self.three_hop_loss.append(three_hop_loss)

        if ignore_neighbors:
            return one_hop_loss + two_hop_loss - angle_loss
        else:
            return one_hop_loss + two_hop_loss - angle_loss + three_hop_loss - dihedral_loss

    def run_writer_mle(self, L1, molecule_loss, pos_mask_L2):
        """
        Set individual loss values for the batch

        :param L1: true if L1 loss is larger else False
        :param molecule_loss: total loss per molecule (n_true_confs, n_model_confs, batch)
        :param pos_mask_L2: mask defining which ground truth conformers are present (n_true_confs, batch)
        :return:
        """

        one_hop_loss = torch.stack(self.one_hop_loss).view(self.n_true_confs, self.n_model_confs, -1)
        two_hop_loss = torch.stack(self.two_hop_loss).view(self.n_true_confs, self.n_model_confs, -1)
        angle_loss = torch.stack(self.angle_loss).view(self.n_true_confs, self.n_model_confs, -1)
        three_hop_loss = torch.stack(self.three_hop_loss).view(self.n_true_confs, self.n_model_confs, -1)
        dihedral_loss = torch.stack(self.dihedral_loss).view(self.n_true_confs, self.n_model_confs, -1)

        if L1:
            inds = torch.min(molecule_loss, dim=0, keepdim=True).indices

            if DEBUG_NEIGHBORHOOD_PAIRS or self.teacher_force:
                self.one_hop_loss_write = torch.sum(
                    one_hop_loss.gather(0, inds).squeeze(1) * pos_mask_L2) / pos_mask_L2.sum()
                self.two_hop_loss_write = torch.sum(
                    two_hop_loss.gather(0, inds).squeeze(1) * pos_mask_L2) / pos_mask_L2.sum()
                self.angle_loss_write = torch.sum(
                    angle_loss.gather(0, inds).squeeze(1) * pos_mask_L2) / pos_mask_L2.sum()
                self.three_hop_loss_write = torch.sum(
                    three_hop_loss.gather(0, inds).squeeze(1) * pos_mask_L2) / pos_mask_L2.sum()
                self.dihedral_loss_write = torch.sum(
                    dihedral_loss.gather(0, inds).squeeze(1) * pos_mask_L2) / pos_mask_L2.sum()
            else:
                self.one_hop_loss_write = one_hop_loss.gather(0, inds).mean()
                self.two_hop_loss_write = two_hop_loss.gather(0, inds).mean()
                self.angle_loss_write = angle_loss.gather(0, inds).mean()
                self.three_hop_loss_write = three_hop_loss.gather(0, inds).mean()
                self.dihedral_loss_write = dihedral_loss.gather(0, inds).mean()

        else:
            inds = torch.min(molecule_loss, dim=1, keepdim=True).indices

            self.one_hop_loss_write = torch.sum(
                one_hop_loss.gather(1, inds).squeeze(1) * pos_mask_L2) / pos_mask_L2.sum()
            self.two_hop_loss_write = torch.sum(
                two_hop_loss.gather(1, inds).squeeze(1) * pos_mask_L2) / pos_mask_L2.sum()
            self.angle_loss_write = torch.sum(angle_loss.gather(1, inds).squeeze(1) * pos_mask_L2) / pos_mask_L2.sum()
            self.three_hop_loss_write = torch.sum(
                three_hop_loss.gather(1, inds).squeeze(1) * pos_mask_L2) / pos_mask_L2.sum()
            self.dihedral_loss_write = torch.sum(
                dihedral_loss.gather(1, inds).squeeze(1) * pos_mask_L2) / pos_mask_L2.sum()

        # reset
        self.one_hop_loss = []
        self.two_hop_loss = []
        self.angle_loss = []
        self.dihedral_loss = []
        self.three_hop_loss = []

    def run_writer_ot_emd(self, ot_mat_list, n_true_confs_batch):
        """
        Set individual loss values for the batch

        :param ot_mat_list: list of optimal transport solution matrices (len batch with shape (n_true_confs for the
            molecule, n_model_confs))
        :param n_true_confs_batch: number of true conformers per molecule (batch)
        :return:
        """

        one_hop_loss = torch.stack(self.one_hop_loss).view(self.n_true_confs, self.n_model_confs, -1)
        two_hop_loss = torch.stack(self.two_hop_loss).view(self.n_true_confs, self.n_model_confs, -1)
        angle_loss = torch.stack(self.angle_loss).view(self.n_true_confs, self.n_model_confs, -1)
        three_hop_loss = torch.stack(self.three_hop_loss).view(self.n_true_confs, self.n_model_confs, -1)
        dihedral_loss = torch.stack(self.dihedral_loss).view(self.n_true_confs, self.n_model_confs, -1)

        self.one_hop_loss_write = 0
        self.two_hop_loss_write = 0
        self.angle_loss_write = 0
        self.three_hop_loss_write = 0
        self.dihedral_loss_write = 0

        for i, ot_mat in enumerate(ot_mat_list):

            if self.teacher_force:
                self.one_hop_loss_write += torch.sum(
                    ot_mat * one_hop_loss[:n_true_confs_batch[i], :n_true_confs_batch[i], i]) / len(ot_mat_list)
                self.two_hop_loss_write += torch.sum(
                    ot_mat * two_hop_loss[:n_true_confs_batch[i], :n_true_confs_batch[i], i]) / len(ot_mat_list)
                self.angle_loss_write += torch.sum(
                    ot_mat * angle_loss[:n_true_confs_batch[i], :n_true_confs_batch[i], i]) / len(ot_mat_list)
                self.three_hop_loss_write += torch.sum(
                    ot_mat * three_hop_loss[:n_true_confs_batch[i], :n_true_confs_batch[i], i]) / len(ot_mat_list)
                self.dihedral_loss_write += torch.sum(
                    ot_mat * dihedral_loss[:n_true_confs_batch[i], :n_true_confs_batch[i], i]) / len(ot_mat_list)
            else:
                self.one_hop_loss_write += torch.sum(ot_mat * one_hop_loss[:n_true_confs_batch[i], :, i]) / len(
                    ot_mat_list)
                self.two_hop_loss_write += torch.sum(ot_mat * two_hop_loss[:n_true_confs_batch[i], :, i]) / len(
                    ot_mat_list)
                self.angle_loss_write += torch.sum(ot_mat * angle_loss[:n_true_confs_batch[i], :, i]) / len(ot_mat_list)
                self.three_hop_loss_write += torch.sum(ot_mat * three_hop_loss[:n_true_confs_batch[i], :, i]) / len(
                    ot_mat_list)
                self.dihedral_loss_write += torch.sum(ot_mat * dihedral_loss[:n_true_confs_batch[i], :, i]) / len(
                    ot_mat_list)

        # reset
        self.one_hop_loss = []
        self.two_hop_loss = []
        self.angle_loss = []
        self.dihedral_loss = []
        self.three_hop_loss = []
