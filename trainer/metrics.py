import torch
from pytorch_lightning.metrics.utils import reduce
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
import numpy as np

from pytorch_lightning.metrics import F1

from commons.losses import cov_loss, std_loss, uniformity_loss
from datasets.qm9_dataset import QM9Dataset


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
    def __init__(self, dataset: QM9Dataset):
        super().__init__()
        self.means = dataset.targets_mean
        self.stds = dataset.targets_std
        self.eV2meV = dataset.eV2meV

    def forward(self, preds, targets):
        preds = denormalize(preds, self.means, self.stds, self.eV2meV)
        targets = denormalize(targets, self.means, self.stds, self.eV2meV)
        loss = F.l1_loss(preds, targets)
        return loss

class MAE(nn.Module):
    def __init__(self,):
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
    def __init__(self, dataset: QM9Dataset):
        super().__init__()
        self.means = dataset.targets_mean
        self.stds = dataset.targets_std
        self.eV2meV = dataset.eV2meV

    def forward(self, preds, targets):
        preds = denormalize(preds, self.means, self.stds, self.eV2meV)
        targets = denormalize(targets, self.means, self.stds, self.eV2meV)
        return F.mse_loss(preds, targets)


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
        return std_loss(x1) + std_loss(x2)


class Alignment(nn.Module):
    def __init__(self, alpha=2) -> None:
        super(Alignment, self).__init__()
        self.alpha = alpha

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        return (x1 - x2).norm(dim=1).pow(self.alpha).mean()


class Uniformity(nn.Module):
    def __init__(self, t=2) -> None:
        super(Uniformity, self).__init__()
        self.t = t

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        return uniformity_loss(x1,x2)


class TruePositiveRate(nn.Module):
    def __init__(self, threshold=0.5) -> None:
        super(TruePositiveRate, self).__init__()
        self.threshold = threshold

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = x1.size()
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


class F1Contrastive(nn.Module):
    def __init__(self, threshold=0.5, device='cuda') -> None:
        super(F1Contrastive, self).__init__()
        self.f1 = F1(num_classes=1, average='weighted', threshold=threshold).to(device)

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = x1.size()
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)
        preds = (sim_matrix.view(-1) + 1) / 2
        if pos_mask != None:  # if we are comparing local with global
            targets = pos_mask.view(-1)
        else:  # if we are comparing global with global
            targets = torch.eye(batch_size, device=x1.device).view(-1)
        return self.f1(preds, targets.long())


class PositiveSimilarity(nn.Module):
    """
        https://en.wikipedia.org/wiki/Cosine_similarity
    """

    def __init__(self) -> None:
        super(PositiveSimilarity, self).__init__()

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
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


class NegativeSimilarity(nn.Module):
    def __init__(self) -> None:
        super(NegativeSimilarity, self).__init__()

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = x1.size()
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


class MetricFunctionToClass(nn.Module):
    def __init__(self, function, name=None, **kwargs):
        super().__init__()
        self.name = name if name is not None else function.__name__
        self.function = function
        self.kwargs = kwargs

    def forward(self,
                preds: torch.Tensor,
                target: torch.Tensor, ):
        return self.function(preds=preds, target=target, **self.kwargs)


def pearsonr(
        preds: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'elementwise_mean'
) -> torch.Tensor:
    """
    Computes the pearsonr correlation.
    
    Arguments
    ------------
        preds: estimated labels
        target: ground truth labels
        reduction: a method to reduce metric score over labels.
            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Returns
    -------------
        Tensor with the pearsonr

    Example:
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> pearsonr(x, y)
        tensor(0.9439)
    """

    shifted_x = preds - torch.mean(preds, dim=0)
    shifted_y = target - torch.mean(target, dim=0)
    sigma_x = torch.sqrt(torch.sum(shifted_x ** 2, dim=0))
    sigma_y = torch.sqrt(torch.sum(shifted_y ** 2, dim=0))

    pearson = torch.sum(shifted_x * shifted_y, dim=0) / (sigma_x * sigma_y + 1.0e-10)
    pearson = torch.clamp(pearson, min=-1, max=1)
    pearson = reduce(pearson, reduction=reduction)
    return pearson


def spearmanr(
        preds: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'elementwise_mean'
) -> torch.Tensor:
    """
    Computes the spearmanr correlation.
    
    Arguments
    ------------
        preds: estimated labels
        target: ground truth labels
        reduction: a method to reduce metric score over labels.
            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Returns
    -------------
        Tensor with the spearmanr

    Example:
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 1.5])
        >>> spearmanr(x, y)
        tensor(0.8)
    """

    pred_rank = torch.argsort(preds, dim=0).float()
    target_rank = torch.argsort(target, dim=0).float()
    spearman = pearsonr(pred_rank, target_rank, reduction=reduction)
    return spearman
