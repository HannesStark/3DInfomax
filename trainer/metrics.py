from typing import Union

import torch
from ogb.graphproppred import Evaluator
from ogb.lsc import PCQM4MEvaluator
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn

from commons.losses import cov_loss, uniformity_loss
from datasets.geom_drugs_dataset import GEOMDrugs
from datasets.qm9_dataset import QM9Dataset


class PearsonR(nn.Module):
    """
    Takes a single target property of the QM9 dataset, denormalizes it and turns in into meV from eV if it  is an energy
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        shifted_x = preds - torch.mean(preds, dim=0)
        shifted_y = targets - torch.mean(targets, dim=0)
        sigma_x = torch.sqrt(torch.sum(shifted_x ** 2, dim=0))
        sigma_y = torch.sqrt(torch.sum(shifted_y ** 2, dim=0))

        pearson = torch.sum(shifted_x * shifted_y, dim=0) / (sigma_x * sigma_y + 1e-8)
        pearson = torch.clamp(pearson, min=-1, max=1)
        pearson = pearson.mean()
        return pearson


class QM9SingleTargetDenormalizedL1(nn.Module):
    """
    Takes a single target property of the QM9 dataset, denormalizes it and turns in into meV from eV if it  is an energy
    """

    def __init__(self, dataset: QM9Dataset, task: str):
        super().__init__()
        self.task_index = dataset.target_tasks.index(task)  # what index the task has in the target tensor
        self.means = dataset.targets_mean
        self.stds = dataset.targets_std
        self.eV2meV = dataset.eV2meV


    def forward(self, preds, targets):
        preds = denormalize(preds, self.means, self.stds, self.eV2meV)
        targets = denormalize(targets, self.means, self.stds, self.eV2meV)
        pred = preds[:, self.task_index]
        target = targets[:, self.task_index]
        loss = F.l1_loss(pred, target)
        return loss


class QM9DenormalizedL1(nn.Module):
    def __init__(self, dataset: Union[QM9Dataset, GEOMDrugs]):
        super().__init__()
        self.means = dataset.targets_mean
        self.stds = dataset.targets_std
        self.eV2meV = None
        if isinstance(dataset, QM9Dataset):
            self.eV2meV = dataset.eV2meV

    def forward(self, preds, targets):
        preds = denormalize(preds, self.means, self.stds, self.eV2meV)
        targets = denormalize(targets, self.means, self.stds, self.eV2meV)
        loss = F.l1_loss(preds, targets)
        return loss


class MAE(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, preds, targets):
        loss = F.l1_loss(preds, targets)
        return loss


def denormalize(normalized: torch.tensor, means, stds, eV2meV):
    denormalized = normalized * stds[None, :] + means[None, :]  # [batchsize, n_tasks]
    if eV2meV:
        denormalized = denormalized * eV2meV[None, :]
    return denormalized


class QM9DenormalizedL2(nn.Module):
    def __init__(self, dataset: Union[QM9Dataset, GEOMDrugs]):
        super().__init__()
        self.means = dataset.targets_mean
        self.stds = dataset.targets_std
        self.eV2meV = None
        if isinstance(dataset, QM9Dataset):
            self.eV2meV = dataset.eV2meV

    def forward(self, preds, targets):
        preds = denormalize(preds, self.means, self.stds, self.eV2meV)
        targets = denormalize(targets, self.means, self.stds, self.eV2meV)
        return F.mse_loss(preds, targets)


class OGBEvaluator(nn.Module):
    def __init__(self, d_name, metric='rocauc'):
        super().__init__()
        self.evaluator = Evaluator(name=d_name)
        self.val_only = metric == 'rocauc'
        self.metric = metric

    def forward(self, preds, targets):
        if preds.shape[1] != self.evaluator.num_tasks:
            return torch.tensor(float('NaN'))
        input_dict = {"y_true": targets, "y_pred": preds}
        return torch.tensor(self.evaluator.eval(input_dict)[self.metric])


class PCQM4MEvaluatorWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.evaluator = PCQM4MEvaluator()
        self.val_only = False

    def forward(self, preds, targets):
        if preds.shape[1] != 1:
            return torch.tensor(float('NaN'))
        input_dict = {"y_true": targets.long().squeeze(), "y_pred": preds.squeeze()}
        return torch.tensor(self.evaluator.eval(input_dict)['mae'])


class Rsquared(nn.Module):
    """
        Coefficient of determination/ R squared measure tells us the goodness of fit of our model.
        Rsquared = 1 means that the regression predictions perfectly fit the data.
        If Rsquared is less than 0 then our model is worse than the mean predictor.
        https://en.wikipedia.org/wiki/Coefficient_of_determination
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        total_SS = ((targets - targets.mean()) ** 2).sum()
        residual_SS = ((targets - preds) ** 2).sum()
        return 1 - residual_SS / total_SS


class MeanPredictorLoss(nn.Module):
    """
        https://en.wikipedia.org/wiki/Cosine_similarity
    """

    def __init__(self, loss_func) -> None:
        super(MeanPredictorLoss, self).__init__()
        self.loss_func = loss_func

    def forward(self, x1: Tensor, targets: Tensor) -> Tensor:
        return self.loss_func(torch.full_like(targets, targets.mean()), targets)


class DimensionCovariance(nn.Module):
    def __init__(self) -> None:
        super(DimensionCovariance, self).__init__()

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        return cov_loss(x1) + cov_loss(x2)


class BatchVariance(nn.Module):
    def __init__(self) -> None:
        super(BatchVariance, self).__init__()

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        return x1.std(dim=0).mean() + x2.std(dim=0).mean()


class Conformer3DVariance(nn.Module):
    def __init__(self, normalize=False) -> None:
        super(Conformer3DVariance, self).__init__()
        self.norm = normalize

    def forward(self, z1: Tensor, z2: Tensor, pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = z1.size()
        _, metric_dim = z2.size()

        z1 = z1.view(batch_size, 2, metric_dim)
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        if self.norm:
            z2 = F.normalize(z2, dim=2)

        z2_vars = z2.var(1)  # [batch_size, metric_dim]
        return z2_vars.mean()


class Conformer2DVariance(nn.Module):
    def __init__(self, normalize=False) -> None:
        super(Conformer2DVariance, self).__init__()
        self.norm = normalize

    def forward(self, z1: Tensor, z2: Tensor, pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = z1.size()
        _, metric_dim = z2.size()

        z1 = z1.view(batch_size, 2, metric_dim)
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        if self.norm:
            z1 = F.normalize(z1, dim=2)
        z1_vars = torch.exp(z1[:, 1, :])  # [batch_size, metric_dim]
        return z1_vars.mean()


class Alignment(nn.Module):
    def __init__(self, alpha=2) -> None:
        super(Alignment, self).__init__()
        self.alpha = alpha

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        if x1.shape != x2.shape and pos_mask == None:  # if we have noisy samples our x2 has them appended at the end so we just take the non noised ones to calculate the similaritiy
            x2 = x2[:len(x1)]
        return (x1 - x2).norm(dim=1).pow(self.alpha).mean()


class Uniformity(nn.Module):
    def __init__(self, t=2) -> None:
        super(Uniformity, self).__init__()
        self.t = t

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        return uniformity_loss(x1, x2)


class TruePositiveRate(nn.Module):
    def __init__(self, threshold=0.5) -> None:
        super(TruePositiveRate, self).__init__()
        self.threshold = threshold

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = x1.size()
        if x1.shape != x2.shape and pos_mask == None:  # if we have noisy samples our x2 has them appended at the end so we just take the non noised ones to calculate the similaritiy
            x2 = x2[:batch_size]
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)

        preds: Tensor = (sim_matrix + 1) / 2 > self.threshold
        if pos_mask == None:  # if we are comparing global with global
            pos_mask = torch.eye(batch_size, device=x1.device)

        num_positives = len(x1)
        true_positives = num_positives - ((preds.long() - pos_mask) * pos_mask).count_nonzero()

        return true_positives / num_positives


class TrueNegativeRate(nn.Module):
    def __init__(self, threshold=0.5) -> None:
        super(TrueNegativeRate, self).__init__()
        self.threshold = threshold

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = x1.size()
        if x1.shape != x2.shape and pos_mask == None:  # if we have noisy samples our x2 has them appended at the end so we just take the non noised ones to calculate the similaritiy
            x2 = x2[:batch_size]
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)

        preds: Tensor = (sim_matrix + 1) / 2 > self.threshold
        if pos_mask == None:  # if we are comparing global with global
            pos_mask = torch.eye(batch_size, device=x1.device)
        neg_mask = 1 - pos_mask

        num_negatives = len(x1) * (len(x2) - 1)
        true_negatives = num_negatives - (((~preds).long() - neg_mask) * neg_mask).count_nonzero()

        return true_negatives / num_negatives


class ContrastiveAccuracy(nn.Module):
    def __init__(self, threshold=0.5) -> None:
        super(ContrastiveAccuracy, self).__init__()
        self.threshold = threshold

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = x1.size()
        if x1.shape != x2.shape and pos_mask == None:  # if we have noisy samples our x2 has them appended at the end so we just take the non noised ones to calculate the similaritiy
            x2 = x2[:batch_size]
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)

        preds: Tensor = (sim_matrix + 1) / 2 > self.threshold
        if pos_mask == None:  # if we are comparing global with global
            pos_mask = torch.eye(batch_size, device=x1.device)
        neg_mask = 1 - pos_mask

        num_positives = len(x1)
        num_negatives = len(x1) * (len(x2) - 1)
        true_positives = num_positives - ((preds.long() - pos_mask) * pos_mask).count_nonzero()
        true_negatives = num_negatives - (((~preds).long() - neg_mask) * neg_mask).count_nonzero()
        return (true_positives / num_positives + true_negatives / num_negatives) / 2


class PositiveSimilarity(nn.Module):
    """
        https://en.wikipedia.org/wiki/Cosine_similarity
    """

    def __init__(self) -> None:
        super(PositiveSimilarity, self).__init__()

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        if x1.shape != x2.shape and pos_mask == None:  # if we have noisy samples our x2 has them appended at the end so we just take the non noised ones to calculate the similaritiy
            x2 = x2[:len(x1)]

        if pos_mask != None:  # if we are comparing local with global
            batch_size, _ = x1.size()
            sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

            x1_abs = x1.norm(dim=1)
            x2_abs = x2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)
            pos_sim = (pos_mask * sim_matrix).sum(dim=1)
        else:  # if we are comparing global with global
            pos_sim = F.cosine_similarity(x1, x2)
        pos_sim = (pos_sim + 1) / 2
        return pos_sim.mean(dim=0)


class PositiveProb(nn.Module):
    def __init__(self) -> None:
        super(PositiveProb, self).__init__()

    def forward(self, z1: Tensor, z2: Tensor, pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = z1.size()
        _, metric_dim = z2.size()

        if batch_size == metric_dim == 2:  # for the dictionary init in the beginning in the trainer
            return torch.tensor(float('Nan'))

        z1 = z1.view(batch_size, 2, metric_dim)
        z1_means = z1[:, 0, :]  # [batch_size, metric_dim]
        z1_stds = torch.exp(z1[:, 1, :] / 2)  # [batch_size, metric_dim]
        z2 = z2.view(-1, batch_size, metric_dim).permute(1, 0, 2)  # [batch_size, num_conformers, metric_dim]

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
        pos_sim = torch.diagonal(likelihood_kernel)
        return pos_sim.mean(dim=0)


class NegativeProb(nn.Module):
    def __init__(self) -> None:
        super(NegativeProb, self).__init__()

    def forward(self, z1: Tensor, z2: Tensor, pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = z1.size()
        _, metric_dim = z2.size()

        if batch_size == metric_dim == 2:  # for the dictionary init in the beginning in the trainer
            return torch.tensor(float('Nan'))

        z1 = z1.view(batch_size, 2, metric_dim)
        z1_means = z1[:, 0, :]  # [batch_size, metric_dim]
        z1_stds = torch.exp(z1[:, 1, :] / 2)  # [batch_size, metric_dim]
        z2 = z2.view(-1, batch_size, metric_dim).permute(1, 0, 2)  # [batch_size, num_conformers, metric_dim]

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
        neg_sim = (likelihood_kernel.sum(dim=1) - torch.diagonal(likelihood_kernel))
        return neg_sim.mean(dim=0)


class PositiveSimilarityMultiplePositivesSeparate2d(nn.Module):
    """
        https://en.wikipedia.org/wiki/Cosine_similarity
    """

    def __init__(self) -> None:
        super(PositiveSimilarityMultiplePositivesSeparate2d, self).__init__()

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        batch_size, _ = z1.size()
        _, metric_dim = z2.size()
        z1 = z1.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        # only take the direct similarities such that one 2D representation is similar to one 3d conformer
        pos_sim = (z1 * z2).sum(dim=2)  # [batch_size, num_conformers]

        z1_abs = z1.norm(dim=2)
        z2_abs = z2.norm(dim=2)
        pos_sim /= (z1_abs * z2_abs)  # [batch_size, num_conformers]
        pos_sim = (pos_sim.sum(dim=1) + 1) / 2
        return pos_sim.mean(dim=0)


class NegativeSimilarityMultiplePositivesSeparate2d(nn.Module):
    def __init__(self) -> None:
        super(NegativeSimilarityMultiplePositivesSeparate2d, self).__init__()

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        batch_size, num_conformers_times_metric_dim = z1.size()
        _, metric_dim = z2.size()
        num_conformers = num_conformers_times_metric_dim / metric_dim
        z1 = z1.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        sim_matrix = torch.einsum('ilk,juk->ijlu', z1, z2)  # [batch_size, batch_size, num_conformers]

        z1_abs = z1.norm(dim=2)
        z2_abs = z2.norm(dim=2)
        sim_matrix = sim_matrix / torch.einsum('il,ju->ijlu', z1_abs, z2_abs)

        sim_matrix = sim_matrix.reshape(batch_size, batch_size, -1).sum(dim=2)  # [batch_size, batch_size]
        neg_sim = (sim_matrix.sum(dim=1) - torch.diagonal(sim_matrix)) / (num_conformers ** 2 * (batch_size - 1))
        neg_sim = (neg_sim + 1) / 2
        return neg_sim.mean(dim=0)


class NegativeSimilarity(nn.Module):
    def __init__(self) -> None:
        super(NegativeSimilarity, self).__init__()

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = x1.size()
        if x1.shape != x2.shape and pos_mask == None:  # if we have noisy samples our x2 has them appended at the end so we just take the non noised ones to calculate the similaritiy
            x2 = x2[:batch_size]
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)

        if pos_mask != None:  # if we are comparing local with global
            pos_sim = (pos_mask * sim_matrix).sum(dim=1)
        else:  # if we are comparing global with global
            pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        neg_sim = (sim_matrix.sum(dim=1) - pos_sim) / (batch_size - 1)
        neg_sim = (neg_sim + 1) / 2
        return neg_sim.mean(dim=0)
