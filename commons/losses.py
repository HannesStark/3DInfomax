import itertools
import math

import dgl
import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal
from torch.nn.modules.loss import _Loss, L1Loss, MSELoss, BCEWithLogitsLoss
import numpy as np
import torch.nn.functional as F


class OGBNanLabelBCEWithLogitsLoss(_Loss):
    def __init__(self) -> None:
        super(OGBNanLabelBCEWithLogitsLoss, self).__init__()
        self.bce_loss = BCEWithLogitsLoss()
    def forward(self, pred, target, **kwargs):
        is_labeled = ~torch.isnan(target)

        loss = self.bce_loss(pred[is_labeled], target[is_labeled])
        return loss

class OGBNanLabelMSELoss(_Loss):
    def __init__(self) -> None:
        super(OGBNanLabelMSELoss, self).__init__()
        self.mse_loss = MSELoss()
    def forward(self, pred, target, **kwargs):
        is_labeled = ~torch.isnan(target)

        loss = self.mse_loss(pred[is_labeled], target[is_labeled])
        return loss

class CriticLoss(_Loss):
    def __init__(self) -> None:
        super(CriticLoss, self).__init__()

    def forward(self, z2, reconstruction, **kwargs):
        batch_size, metric_dim, repeats = reconstruction.size()
        z2_norm = F.normalize(z2, dim=1, p=2)[..., None].repeat(1, 1, repeats)
        reconstruction_norm = F.normalize(reconstruction, dim=1, p=2)
        loss = (((z2_norm - reconstruction_norm) ** 2).sum(dim=1)).mean()
        return loss


class BarlowTwinsLoss(_Loss):
    def __init__(self, scale_loss=1 / 32, lambd=3.9e-3, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        "Loss function from the Barlow twins paper from yann lecun"
        super(BarlowTwinsLoss, self).__init__()
        self.scale_loss = scale_loss
        self.lambd = lambd
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, **kwargs) -> Tensor:
        batch_size, metric_dim = z1.size()
        z1_norm = (z1 - z1.mean(dim=0)) / z1.std(dim=0)  # [batch_size, metric_dim]
        z2_norm = (z2 - z2.mean(dim=0)) / z2.std(dim=0)  # [batch_size, metric_dim]
        corr_matrix = (z1_norm.T @ z2_norm) / batch_size  # [metric_dim, metric_dim]

        on_diag = torch.diagonal(corr_matrix).add_(-1).pow(2).sum().mul(self.scale_loss)

        off_diag = corr_matrix.flatten()[:-1].view(metric_dim - 1, metric_dim + 1)[:, 1:].flatten()
        off_diag = off_diag.pow(2).sum().mul(self.scale_loss)

        loss = on_diag + self.lambd * off_diag
        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class CosineSimilarityLoss(_Loss):
    def __init__(self, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super(CosineSimilarityLoss, self).__init__()
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        # see the "Bootstrap your own latent" paper equation 2 for the loss"
        # this loss is equivalent to 2 - 2*cosine_similarity
        x = F.normalize(z1, dim=-1, p=2)
        y = F.normalize(z2, dim=-1, p=2)
        loss = (((x - y) ** 2).sum(dim=-1)).mean()
        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class RegularizationLoss(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            lambd: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, uniformity_reg=0, variance_reg=1, covariance_reg=0.04) -> None:
        super(RegularizationLoss, self).__init__()
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        loss = self.mse_loss(z1, z2)
        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class NTXent(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super(NTXent, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / (torch.einsum('i,j->ij', z1_abs, z2_abs) + 1e-8)

        sim_matrix = torch.exp(sim_matrix / self.tau)
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss

class NTXentAE(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0, reconstruction_reg = 1) -> None:
        super(NTXentAE, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg
        self.mse_loss = MSELoss()
        self.reconstruction_reg = reconstruction_reg

    def forward(self, z1, z2, distances, distance_pred, **kwargs):
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / (torch.einsum('i,j->ij', z1_abs, z2_abs) + 1e-8)

        sim_matrix = torch.exp(sim_matrix / self.tau)
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss, self.reconstruction_reg * self.mse_loss(distances, distance_pred)

class NTXentMultiplePositives(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0,
                 conformer_variance_reg=0) -> None:
        super(NTXentMultiplePositives, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg
        self.conformer_variance_reg = conformer_variance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        '''
        :param z1: batchsize, metric dim
        :param z2: batchsize*num_conformers, metric dim
        '''
        batch_size, metric_dim = z1.size()
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]

        sim_matrix = torch.einsum('ik,juk->iju', z1, z2)  # [batch_size, batch_size, num_conformers]

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=2)
            sim_matrix = sim_matrix / torch.einsum('i,ju->iju', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / self.tau)  # [batch_size, batch_size, num_conformers]

        sim_matrix = sim_matrix.sum(dim=2)  # [batch_size, batch_size]
        pos_sim = torch.diagonal(sim_matrix)  # [batch_size]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.conformer_variance_reg > 0:
            std = torch.sqrt(z2.var(dim=1) + 1e-04)
            std_conf_loss = torch.mean(torch.relu(1 - std))
            loss += self.conformer_variance_reg * std_conf_loss
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class KLDivergenceMultiplePositives(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = False, tau: float = 0.5, uniformity_reg=0, variance_reg=0,
                 covariance_reg=0) -> None:
        super(KLDivergenceMultiplePositives, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        '''
        :param z1: batchsize, metric dim*2
        :param z2: batchsize*num_conformers, metric dim
        '''
        batch_size, _ = z1.size()
        _, metric_dim = z2.size()

        z1 = z1.view(batch_size, 2, metric_dim)
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        if self.norm:
            z1 = F.normalize(z1, dim=2)
            z2 = F.normalize(z2, dim=2)
        z1_means = z1[:, 0, :]  # [batch_size, metric_dim]
        z1_vars = torch.exp(z1[:, 1, :])  # [batch_size, metric_dim]
        z2_means = z2.mean(1)  # [batch_size, metric_dim]
        z2_vars = z2.var(1) + 1e-6  # [batch_size, metric_dim]
        try:
            normal1 = MultivariateNormal(z1_means, torch.diag_embed(z1_vars))
        except:
            print(z1_vars)
        try:
            normal2 = MultivariateNormal(z2_means, torch.diag_embed(z2_vars))
        except:
            print(z2_vars)
        # kl_div = torch.distributions.kl_divergence(normal1, normal2)
        kl_div = torch.distributions.kl_divergence(normal2, normal1)
        loss = kl_div.mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class JSDMultiplePositivesLoss(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super(JSDMultiplePositivesLoss, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        '''
        :param z1: batchsize, metric dim*2
        :param z2: batchsize*num_conformers, metric dim
        '''
        batch_size, _ = z1.size()
        _, metric_dim = z2.size()

        z1 = z1.view(batch_size, 2, metric_dim)
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        if self.norm:
            z1 = F.normalize(z1, dim=2)
            z2 = F.normalize(z2, dim=2)

        z1_means = z1[:, 0, :].unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, batch_size, metric_dim]
        z1_vars = (torch.exp(z1[:, 1, :])).unsqueeze(0).expand(batch_size, -1,
                                                               -1)  # [batch_size, batch_size, metric_dim]
        z2_means = z2.mean(1).unsqueeze(1).expand(-1, batch_size, -1)  # [batch_size, batch_size, metric_dim]
        z2_vars = z2.var(1).unsqueeze(1).expand(-1, batch_size, -1)  # [batch_size, batch_size, metric_dim]

        normal1 = MultivariateNormal(z1_means, torch.diag_embed(z1_vars))
        normal2 = MultivariateNormal(z2_means, torch.diag_embed(z2_vars))
        # kl_div = torch.distributions.kl_divergence(normal1, normal2)
        kl_div = -torch.distributions.kl_divergence(normal2, normal1)

        sigma_alpha = 1 / (0.5 / z1_vars + 0.5 / z2_vars)
        mu_alpha = sigma_alpha * (z1_means * 0.5 / z1_vars + 0.5 * z2_means / z2_vars)

        log_det_diff = torch.log((z2_vars.prod(dim=2) + 1e-5) / (z1_vars.prod(dim=2)))
        trace_inv = ((1 / (z2_vars + 1e-5)) * z1_vars).sum(dim=2)
        mean_sigma_mean = ((z2_means - z1_means) ** 2 * (1 / (z2_vars + 1e-5))).sum(dim=2)
        kl_similarity2 = 0.5 * (log_det_diff - metric_dim + trace_inv + mean_sigma_mean)
        kl_similarity = []
        for i, z1_mean in enumerate(z1_means[0, :, :]):
            for j, z2_mean in enumerate(z2_means[:, 0, :]):
                z1_var = z1_vars[0, :, :][i]  # [metric_dim]
                z2_var = z2_vars[:, 0, :][j]  # [metric_dim]
                log_det_diff = torch.log(((z2_var).prod() + 1e-5) / ((z1_var).prod() + 1e-5))
                trace_inv = ((1 / (z2_var + 1e-5)) * z1_var).sum()
                mean_sigma_mean = ((z2_mean - z1_mean) ** 2 * (1 / (z2_var + 1e-5))).sum()
                kl_divergence = 0.5 * (log_det_diff - metric_dim + trace_inv + mean_sigma_mean)
                kl_similarity.append(1 - kl_divergence)
        kl_similarity = torch.stack(kl_similarity)
        kl_similarity = kl_similarity.view(batch_size, batch_size)

        sim_matrix = kl_similarity2
        # sim_matrix = torch.exp(kl_similarity / self.tau)
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class NTXentMMDSeparate2D(_Loss):
    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0,
                 kernel_num=5, kernel_mul=2.0) -> None:
        super(NTXentMMDSeparate2D, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, z1, z2, **kwargs) -> Tensor:
        '''
        :param z1: batchsize, metric dim * num_conformers
        :param z2: batchsize * num_conformers, metric dim
        '''
        batch_size, _ = z1.size()
        _, metric_dim = z2.size()
        z1 = z1.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        _, num_conformers, _ = z1.size()
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]

        if self.norm:
            z1 = F.normalize(z1, dim=2)
            z2 = F.normalize(z2, dim=2)

        z1_vec = z1.clone().unsqueeze(0).expand(batch_size, -1, -1,
                                                -1)  # [batch_size, batch_size, num_conformers, metric_dim]
        z2_vec = z2.clone().unsqueeze(1).expand(-1, batch_size, -1,
                                                -1)  # [batch_size, batch_size, num_conformers, metric_dim]

        n_samples = num_conformers * 2
        total = torch.cat([z1_vec, z2_vec], dim=2)

        total0 = total.unsqueeze(2).expand(-1, -1, int(total.size(2)), int(total.size(2)), int(total.size(
            3)))  # [batch_size, batch_size, num_conformers*2, num_conformers*2, metric_dim]
        total1 = total.unsqueeze(3).expand(-1, -1, int(total.size(2)), int(total.size(2)), int(total.size(
            3)))  # [batch_size, batch_size, num_conformers*2, num_conformers*2, metric_dim]
        L2_distance = ((total0 - total1) ** 2).sum(4)
        bandwidth = torch.sum(L2_distance.data, dim=(2, 3)) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp[:, :, None, None].expand(-1, -1, n_samples, n_samples))
                      for bandwidth_temp in bandwidth_list]
        kernel_val = sum(kernel_val)
        XX = kernel_val[:, :, :num_conformers, :num_conformers]
        YY = kernel_val[:, :, num_conformers:, num_conformers:]
        XY = kernel_val[:, :, :num_conformers, num_conformers:]
        YX = kernel_val[:, :, num_conformers:, :num_conformers]
        mmd_loss = torch.mean(XX + YY - XY - YX, dim=(2, 3))
        mmd_similarity = 1 / (mmd_loss + 1)

        sim_matrix = torch.exp(mmd_similarity / self.tau)

        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class KLDivergenceMultiplePositivesV2(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super(KLDivergenceMultiplePositivesV2, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        '''
        :param z1: batchsize, metric dim*2
        :param z2: batchsize*num_conformers, metric dim
        '''
        batch_size, _ = z1.size()
        _, metric_dim = z2.size()

        z1 = z1.view(batch_size, 2, metric_dim)
        z1_means = z1[:, 0, :]  # [batch_size, metric_dim]
        z1_stds = torch.exp(z1[:, 1, :] / 2)  # [batch_size, metric_dim]
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        z2_means = z2.mean(1)  # [batch_size, metric_dim]
        z2_stds = z2.std(1)  # [batch_size, metric_dim]

        kl_div_kernel = []
        for i, z1_mean in enumerate(z1_means):
            for j, z2_mean in enumerate(z2_means):
                z1_std = z1_stds[i]  # [metric_dim]
                z2_std = z2_stds[j] + 1e-5  # [metric_dim]
                p = torch.distributions.Normal(z1_mean, z1_std)
                q = torch.distributions.Normal(z2_mean, z2_std)
                kl_divergence = torch.distributions.kl.kl_divergence(p, q)
                kl_div_kernel.append(kl_divergence)
        kl_div_kernel = torch.stack(kl_div_kernel)
        kl_div_kernel = kl_div_kernel.view(batch_size, batch_size)

        sim_matrix = torch.exp(kl_div_kernel / self.tau)  # [batch_size, batch_size]
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class NTXentLikelihoodLoss(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0,
                 conformer_variance_reg=0) -> None:
        super(NTXentLikelihoodLoss, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg
        self.conformer_variance_reg = conformer_variance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        '''
        :param z1: batchsize, metric dim*2
        :param z2: batchsize*num_conformers, metric dim
        '''
        batch_size, _ = z1.size()
        _, metric_dim = z2.size()

        z1 = z1.view(batch_size, 2, metric_dim)
        z1_means = z1[:, 0, :]  # [batch_size, metric_dim]
        z1_stds = torch.exp(z1[:, 1, :] / 2)  # [batch_size, metric_dim]
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]

        likelihood_kernel = []
        for i, z1_mean in enumerate(z1_means):
            z1_std = z1_stds[i]  # [metric_dim]
            p = torch.distributions.Normal(z1_mean, z1_std)
            for j, z2_elem in enumerate(z2):
                z2_elem = z2_elem  # [num_conformers, metric_dim]
                prob = torch.exp(p.log_prob(z2_elem))
                likelihood_kernel.append(prob.mean())
        likelihood_kernel = torch.stack(likelihood_kernel)
        likelihood_kernel = likelihood_kernel.view(batch_size, batch_size)

        sim_matrix = torch.exp(likelihood_kernel / self.tau)  # [batch_size, batch_size]
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.conformer_variance_reg > 0:
            std = torch.sqrt(z2.var(dim=1) + 1e-04)
            std_conf_loss = torch.mean(torch.relu(1 - std))
            loss += self.conformer_variance_reg * std_conf_loss
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class NTXentMultiplePositivesV2(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super(NTXentMultiplePositivesV2, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        '''
        :param z1: batch_size, metric dim
        :param z2: batch_size*num_conformers, metric dim
        '''
        batch_size, metric_dim = z1.size()
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]

        pos_sim = (z1[:, None, :] * z2).sum(dim=2)  # [batch_size, num_conformers]
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2[:, 0, :])  # [batch_size, batch_size]
        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=2)
            pos_sim = pos_sim / (z1_abs[:, None] * z2_abs)  # [batch_size, num_conformers]
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs[:, 0])

        sim_matrix = torch.exp(sim_matrix / self.tau)  # [batch_size, batch_size]
        pos_sim = torch.exp(pos_sim / self.tau)  # [batch_size, num_conformers]
        pos_sim = pos_sim.sum(dim=1)  # [batch_size]
        loss = pos_sim / (sim_matrix.sum(dim=1) - torch.diagonal(sim_matrix))
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class NTXentMultiplePositivesV3(_Loss):
    '''
        Just have the multiple positives as extra loss terms over which we take the mean in the end
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super(NTXentMultiplePositivesV3, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        '''
        :param z1: batchsize, metric dim
        :param z2: batchsize*num_conformers, metric dim
        '''
        batch_size, metric_dim = z1.size()
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]

        sim_matrix = torch.einsum('ik,juk->iju', z1, z2)  # [batch_size, batch_size, num_conformers]

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=2)
            sim_matrix = sim_matrix / torch.einsum('i,ju->iju', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / self.tau)  # [batch_size, batch_size, num_conformers]
        pos_sim = sim_matrix[range(batch_size), range(batch_size), :]  # [batch_size, num_conformers]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class NTXentMultiplePositivesSeparate2D(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super(NTXentMultiplePositivesSeparate2D, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        '''
        :param z1: batchsize, metric dim * num_conformers
        :param z2: batchsize * num_conformers, metric dim
        '''
        batch_size, _ = z1.size()
        _, metric_dim = z2.size()
        z1 = z1.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]

        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        sim_matrix = torch.einsum('ilk,juk->ijlu', z1, z2)  # [batch_size, batch_size, num_conformers]

        # only take the direct similarities such that one 2D representation is similar to one 3d conformer
        pos_sim = (z1 * z2).sum(dim=2)  # [batch_size, num_conformers]

        if self.norm:
            z1_abs = z1.norm(dim=2)
            z2_abs = z2.norm(dim=2)
            pos_sim /= (z1_abs * z2_abs)  # [batch_size, num_conformers]
            sim_matrix = sim_matrix / torch.einsum('il,ju->ijlu', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / self.tau)  # [batch_size, batch_size, num_conformers, num_conformers]
        pos_sim = torch.exp(pos_sim / self.tau)  # [batch_size, num_conformers]
        pos_sim = pos_sim.sum(dim=1)

        sim_matrix = sim_matrix.reshape(batch_size, batch_size, -1).sum(dim=2)  # [batch_size, batch_size]
        loss = pos_sim / (sim_matrix.sum(dim=1) - torch.diagonal(sim_matrix))
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class NTXentMinimumMatching(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super(NTXentMinimumMatching, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        '''
        :param z1: batchsize, metric dim * num_conformers
        :param z2: batchsize * num_conformers, metric dim
        '''
        batch_size, _ = z1.size()
        _, metric_dim = z2.size()
        z1 = z1.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]

        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        sim_matrix = torch.einsum('ilk,juk->ijlu', z1, z2)  # [batch_size, batch_size, num_conformers]

        if self.norm:
            z1_abs = z1.norm(dim=2)
            z2_abs = z2.norm(dim=2)
            sim_matrix = sim_matrix / torch.einsum('il,ju->ijlu', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / self.tau)  # [batch_size, batch_size, num_conformers, num_conformers]
        pos_sim = torch.amax(torch.diagonal(sim_matrix, dim1=2, dim2=3), dim=(1, 2))  # [batch_size]
        min_sim_matrix = torch.amin(sim_matrix, dim=(2, 3))  # [batch_size, batch_size]

        loss = pos_sim / (min_sim_matrix.sum(dim=1) - torch.diagonal(min_sim_matrix))
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class MaximumSimilarityMSE(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super(MaximumSimilarityMSE, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg
        self.mse_loss = torch.nn.MSELoss(reduction='none')

    def forward(self, z1, z2, **kwargs) -> Tensor:
        '''
        :param z1: batchsize, metric dim * num_conformers
        :param z2: batchsize * num_conformers, metric dim
        '''
        batch_size, _ = z1.size()
        _, metric_dim = z2.size()
        z1 = z1.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        _, num_conformers, _ = z1.size()

        z1 = z1[:,:,None,:].expand(-1,-1,num_conformers,-1)  # [batch_size, num_conformers, num_conformers, metric_dim]
        z2 = z2[:,None,:,:].expand(-1,num_conformers,-1,-1)   # [batch_size, num_conformers, num_conformers, metric_dim]
        loss = self.mse_loss(z1,z2).mean(dim=-1) # [batch_size, num_conformers, num_conformers]
        loss = torch.amin(loss, dim=(1, 2)).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss

class NTXentMaximumSimilarity(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super(NTXentMaximumSimilarity, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        '''
        :param z1: batchsize, metric dim * num_conformers
        :param z2: batchsize * num_conformers, metric dim
        '''
        batch_size, _ = z1.size()
        _, metric_dim = z2.size()
        z1 = z1.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]

        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        sim_matrix = torch.einsum('ilk,juk->ijlu', z1, z2)  # [batch_size, batch_size, num_conformers]

        if self.norm:
            z1_abs = z1.norm(dim=2)
            z2_abs = z2.norm(dim=2)
            sim_matrix = sim_matrix / torch.einsum('il,ju->ijlu', z1_abs, z2_abs)

        sim_matrix = torch.amax(sim_matrix, dim=(2, 3))  # [batch_size, batch_size]
        sim_matrix = torch.exp(sim_matrix / self.tau)  # [batch_size, batch_size, num_conformers, num_conformers]
        pos_sim = torch.diagonal(sim_matrix)  # [batch_size]

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class NTXentExtraNegatives(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0,
                 extra_negatives_weight=1) -> None:
        super(NTXentExtraNegatives, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg
        self.extra_negatives_weight = extra_negatives_weight

    def forward(self, z1, z2, **kwargs) -> Tensor:
        '''
        :param z1: batchsize, metric dim
        :param z2: batchsize*( 1 + num_extra_negatives), metric dim
        '''
        batch_size, metric_dim = z1.size()
        extra_negatives = z2[batch_size:]  # [batchsize * num_extra_neg, metric_dim]
        z2 = z2[:batch_size]  # [batchsize, metric_dim]

        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)
        extra_negatives = extra_negatives.view(batch_size, -1, metric_dim)  # [batch_size, num_extra_neg, metric_dim]
        sim_extra_negatives = torch.einsum('ik,iuk->iu', z1, extra_negatives)  # [batch_size, num_extra_neg]

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)
            extra_negatives_abs = extra_negatives.norm(dim=-1)  # [batch_size, num_extra_neg]
            sim_extra_negatives = sim_extra_negatives / (extra_negatives_abs * z1_abs[:, None])

        sim_extra_negatives = torch.exp(sim_extra_negatives / self.tau) * self.extra_negatives_weight
        sim_matrix = torch.exp(sim_matrix / self.tau)

        sim_matrix = torch.cat([sim_matrix, sim_extra_negatives], dim=-1)

        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


def uniformity_loss(x1: Tensor, x2: Tensor, t=2) -> Tensor:
    sq_pdist_x1 = torch.pdist(x1, p=2).pow(2)
    uniformity_x1 = sq_pdist_x1.mul(-t).exp().mean().log()
    sq_pdist_x2 = torch.pdist(x2, p=2).pow(2)
    uniformity_x2 = sq_pdist_x2.mul(-t).exp().mean().log()
    return (uniformity_x1 + uniformity_x2) / 2


def cov_loss(x):
    batch_size, metric_dim = x.size()
    x = x - x.mean(dim=0)
    cov = (x.T @ x) / (batch_size - 1)
    off_diag_cov = cov.flatten()[:-1].view(metric_dim - 1, metric_dim + 1)[:, 1:].flatten()
    return off_diag_cov.pow_(2).sum() / metric_dim


def std_loss(x):
    std = torch.sqrt(x.var(dim=0) + 1e-04)
    return torch.mean(torch.relu(1 - std))


class NTXentShuffled(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5) -> None:
        super(NTXentShuffled, self).__init__()
        self.norm = norm
        self.tau = tau

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        z2 = z2[torch.randperm(len(z2))]
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / self.tau)
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


class InfoNCE(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super(InfoNCE, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / self.tau)
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1))
        loss = - torch.log(loss).mean()
        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class InfoNCEHard(_Loss):
    '''
        InfoNCE
        with the adaptation of the 'Contrastive Learning with Hard Negative Samples' paper https://openreview.net/forum?id=CR1XOQ0UTh-
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = False, tau: float = 0.5, tau_plus=0.1, beta=0.5) -> None:
        super(InfoNCEHard, self).__init__()
        self.norm = norm
        self.tau_plus = tau_plus
        self.tau = tau
        self.beta = beta

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / self.tau)
        mask = torch.eye(batch_size, dtype=torch.bool).to(z1.device)
        pos = sim_matrix[mask]
        neg = sim_matrix[~mask].view(batch_size, -1)

        imp = (self.beta * neg.log()).exp()
        reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
        Ng = (-self.tau_plus * (batch_size - 1) * pos + reweight_neg) / (1 - self.tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=(batch_size - 1) * np.e ** (-1 / self.tau))
        loss = -torch.log(pos / (pos + Ng)).mean()
        return loss


class NTXentHard(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        with the adaptation of the 'Contrastive Learning with Hard Negative Samples' paper https://openreview.net/forum?id=CR1XOQ0UTh-
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, tau_plus=0.1, beta=0.1) -> None:
        super(NTXentHard, self).__init__()
        self.norm = norm
        self.tau_plus = tau_plus
        self.tau = tau
        self.beta = beta

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / self.tau)
        mask = torch.eye(batch_size, dtype=torch.bool).to(z1.device)
        pos = sim_matrix[mask]
        neg = sim_matrix[~mask].view(batch_size, -1)

        imp = (self.beta * neg.log()).exp()
        reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
        Ng = (-self.tau_plus * (batch_size - 1) * pos + reweight_neg) / (1 - self.tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=(batch_size - 1) * np.e ** (-1 / self.tau))
        loss = -torch.log(pos / Ng).mean()
        return loss


class NTXentLocalGlobal(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5) -> None:
        super(NTXentLocalGlobal, self).__init__()
        self.norm = norm
        self.tau = tau

    def forward(self, zn, zg, nodes_per_graph) -> Tensor:
        '''
        Args:
            zg: Tensor of shape [n_graphs, z_dim].
            zn: Tensor of shape [n_nodes, z_dim].
            batch: Tensor of shape [n_graphs].
        '''
        num_graphs = zg.shape[0]
        num_nodes = zn.shape[0]
        node_indices = torch.cumsum(nodes_per_graph, dim=0)

        pos_mask = torch.zeros((num_nodes, num_graphs), device=zg.device)
        for graph_idx in range(1, len(node_indices)):
            pos_mask[node_indices[graph_idx - 1]: node_indices[graph_idx], graph_idx] = 1.
        pos_mask[0:node_indices[0], 0] = 1
        neg_mask = 1 - pos_mask

        sim_matrix = torch.einsum('ik,jk->ij', zn, zg)

        if self.norm:
            zn_abs = zn.norm(dim=1)
            zg_abs = zg.norm(dim=1)
            sim_matrix = sim_matrix / (torch.einsum('i,j->ij', zn_abs, zg_abs) + 1e-10)

        sim_matrix = torch.exp(sim_matrix / self.tau)
        pos_sim = (sim_matrix * pos_mask).sum(dim=1)
        neg_sim = (sim_matrix * neg_mask).sum(dim=1)
        loss = pos_sim / (neg_sim)
        loss = - torch.log(loss).mean()

        return loss


class NTXentGlobalLocal(_Loss):
    '''
        this is just the NTXentLocalGlobal with arguments switched around
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, **kwargs) -> None:
        super(NTXentGlobalLocal, self).__init__()
        self.ntxent_local_global = NTXentLocalGlobal(**kwargs)

    def forward(self, zg, zn, nodes_per_graph) -> Tensor:
        '''
        Args:
            zg: Tensor of shape [n_graphs, z_dim].
            zn: Tensor of shape [n_nodes, z_dim].
            batch: Tensor of shape [n_graphs].
        '''

        return self.ntxent_local_global(zn, zg, nodes_per_graph)


class SampleLossWrapper(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, loss_func, fraction_samples=0.1) -> None:
        super(SampleLossWrapper, self).__init__()
        self.loss_func = globals()[loss_func]()
        self.fraction_samples = fraction_samples

    def forward(self, x, y) -> Tensor:
        indices = torch.randint(low=0, high=len(x), size=(int(len(x) * self.fraction_samples),), device=x.device)
        x = torch.index_select(x, dim=0, index=indices)
        y = torch.index_select(y, dim=0, index=indices)
        return self.loss_func(x, y)


class JSELossGlobal(_Loss):

    def __init__(self, **kwargs) -> None:
        super(JSELossGlobal, self).__init__()

    def forward(self, z1, z2, **kwargs) -> Tensor:
        jse = JSE_global_global
        return jse(z1, z2)


class JSELossLocalGlobal(_Loss):

    def __init__(self, **kwargs) -> None:
        super(JSELossLocalGlobal, self).__init__()

    def forward(self, z_n, z_g, graph: dgl.DGLGraph) -> Tensor:
        '''
        Args:
            z_g: Tensor of shape [n_graphs, z_dim].
            z_n: Tensor of shape [n_nodes, z_dim].
            batch: Tensor of shape [n_graphs].
        '''
        # TODO: this doesn not work yet
        raise NotImplementedError('not done')
        device = z_g.device
        num_graphs = z_g.shape[0]
        num_nodes = z_n.shape[0]
        node_indices = torch.cumsum(graph.batch_num_nodes(), dim=0)

        pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
        for graph_idx, nodes in enumerate(node_indices):
            pos_mask[range(nodes, node_indices[graph_idx + 1])][graph_idx] = 1.
        neg_mask = 1 - pos_mask

        d_prime = torch.matmul(z_n, z_g.t())

        E_pos = get_expectation(d_prime * pos_mask, positive=True).sum()
        E_pos = E_pos / num_nodes
        E_neg = get_expectation(d_prime * neg_mask, positive=False).sum()
        E_neg = E_neg / (num_nodes * (num_graphs - 1))
        return E_neg - E_pos


class JSELoss(_Loss):

    def __init__(self, neg_by_crpt=False, **kwargs) -> None:
        super(JSELoss, self).__init__()
        self.neg_by_crpt = neg_by_crpt

    def forward(self, zs, zs_n=None, batch=None, sigma=None) -> Tensor:
        '''
           Args:
               zs: List of tensors of shape [n_views, batch_size, z_dim].
               zs_n: List of tensors of shape [n_views, nodes, z_dim].
               sigma: 2D-array of shape [n_views, n_views] with boolean values.
                   Only required when n_views > 2. If sigma_ij = True, then compute
                   infoNCE between view_i and view_j.
           '''
        if zs_n is not None:
            assert len(zs_n) == len(zs)
            assert batch is not None

            jse = (JSE_local_global_negative_paired
                   if self.neg_by_crpt else JSE_local_global)

            if len(zs) == 1:
                return jse(zs[0], zs_n[0], batch)
            elif len(zs) == 2:
                return (jse(zs[0], zs_n[1], batch) +
                        jse(zs[1], zs_n[0], batch))
            else:
                assert len(zs) == len(sigma)
                loss = 0
                for (i, j) in itertools.combinations(range(len(zs)), 2):
                    if sigma[i][j]:
                        loss += (jse(zs[i], zs_n[j], batch) +
                                 jse(zs[j], zs_n[i], batch))
                return loss

        else:
            jse = JSE_global_global
            if len(zs) == 2:
                return jse(zs[0], zs[1])
            elif len(zs) > 2:
                assert len(zs) == len(sigma)
                loss = 0
                for (i, j) in itertools.combinations(range(len(zs)), 2):
                    if sigma[i][j]:
                        loss += jse(zs[i], zs[j])
                return loss


def JSE_local_global_negative_paired(z_g, z_n, batch):
    '''
    Args:
        z_g: of size [2*n_batch, dim]
        z_n: of size [2*n_batch*nodes_per_batch, dim]
    '''
    device = z_g.device
    num_graphs = int(z_g.shape[0] / 2)  # 4
    num_nodes = int(z_n.shape[0] / 2)  # 4*2000
    z_g, _ = torch.split(z_g, num_graphs)
    z_n, z_n_crpt = torch.split(z_n, num_nodes)

    num_sample_nodes = int(num_nodes / num_graphs)
    z_n = torch.split(z_n, num_sample_nodes)
    z_n_crpt = torch.split(z_n_crpt, num_sample_nodes)

    d_pos = torch.cat([torch.matmul(z_g[i], z_n[i].t()) for i in range(num_graphs)])  # [1, 8000]
    d_neg = torch.cat([torch.matmul(z_g[i], z_n_crpt[i].t()) for i in range(num_graphs)])  # [1, 8000]

    logit = torch.unsqueeze(torch.cat((d_pos, d_neg)), 0)  # [1, 16000]
    lb_pos = torch.ones((1, num_nodes)).to(device)  # [1, 8000]
    lb_neg = torch.zeros((1, num_nodes)).to(device)  # [1, 8000]
    lb = torch.cat((lb_pos, lb_neg), 1)

    b_xent = nn.BCEWithLogitsLoss()
    loss = b_xent(logit, lb) * 0.5  # following mvgrl-node
    return loss


def JSE_local_global(z_g, z_n, batch, measure: str = 'JSD'):
    '''
    Args:
        z_g: Tensor of shape [n_graphs, z_dim].
        z_n: Tensor of shape [n_nodes, z_dim].
        batch: Tensor of shape [n_graphs].
    '''
    device = z_g.device
    num_graphs = z_g.shape[0]
    num_nodes = z_n.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
    neg_mask = torch.ones((num_nodes, num_graphs)).to(device)
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    d_prime = torch.matmul(z_n, z_g.t())

    E_pos = get_positive_expectation(d_prime * pos_mask, measure).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(d_prime * neg_mask, measure).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    return E_neg - E_pos


def JSE_global_global(z1, z2, measure: str = 'JSD'):
    '''
    Args:
        z1, z2: Tensor of shape [batch_size, z_dim].
    '''
    device = z1.device
    num_graphs = z1.shape[0]

    pos_mask = torch.zeros((num_graphs, num_graphs)).to(device)
    neg_mask = torch.ones((num_graphs, num_graphs)).to(device)
    for graphidx in range(num_graphs):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    d_prime = torch.matmul(z1, z2.t())

    E_pos = get_expectation(d_prime * pos_mask, positive=True).sum()
    E_pos = E_pos / num_graphs
    E_neg = get_expectation(d_prime * neg_mask, positive=False).sum()
    E_neg = E_neg / (num_graphs * (num_graphs - 1))
    return E_neg - E_pos


def get_expectation(masked_d_prime, positive=True):
    '''
    Args:
        masked_d_prime: Tensor of shape [n_graphs, n_graphs] for global_global,
                        tensor of shape [n_nodes, n_graphs] for local_global.
        positive (bool): Set True if the d_prime is masked for positive pairs,
                        set False for negative pairs.
    '''
    log_2 = np.log(2.)
    if positive:
        score = log_2 - F.softplus(-masked_d_prime)
    else:
        score = F.softplus(-masked_d_prime) + masked_d_prime - log_2
    return score


def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)  # Note JSD will be shifted
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise Exception('measure does not exist: ', measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2  # Note JSD will be shifted
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples - 1.)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise Exception('measure does not exist: ', measure)

    if average:
        return Eq.mean()
    else:
        return Eq


def log_sum_exp(x, axis=None):
    """Log sum exp function
    Args:
        x: Input.
        axis: Axis over which to perform sum.
    Returns:
        torch.Tensor: log sum exp
    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y
