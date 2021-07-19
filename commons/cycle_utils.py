import torch
import numpy as np


def get_cycle_values(cycle_list, start_at=None):
    start_at = 0 if start_at is None else cycle_list.index(start_at)
    while True:
        yield cycle_list[start_at]
        start_at = (start_at + 1) % len(cycle_list)


def get_cycle_indices(cycle, start_idx):
    cycle_it = get_cycle_values(cycle, start_idx)
    indices = []

    end = 9e99
    start = next(cycle_it)
    a = start
    while start != end:
        b = next(cycle_it)
        indices.append(torch.tensor([a, b]))
        a = b
        end = b

    return indices


def get_current_cycle_indices(cycles, cycle_check, idx):
    c_idx = [i for i, c in enumerate(cycle_check) if c][0]
    current_cycle = cycles.pop(c_idx)
    current_idx = current_cycle[(np.array(current_cycle) == idx.item()).nonzero()[0][0]]
    return get_cycle_indices(current_cycle, current_idx)


def align_coords_Kabsch(p_cycle_coords, q_cycle_coords, p_mask, q_mask=None):
    """
    align p_cycle_coords with q_cycle_coords

    mask indicates which atoms to apply RMSD minimization over; these atoms are used to calculate the
    final rotation and translation matrices, which are applied to ALL atoms
    """
    if not q_mask:
        q_mask = p_mask

    q_cycle_coords_centered = q_cycle_coords[:, q_mask] - q_cycle_coords[:, q_mask].mean(dim=1, keepdim=True)
    p_cycle_coords_centered = p_cycle_coords[:, :, p_mask] - p_cycle_coords[:, :, p_mask].mean(dim=2, keepdim=True)

    H = torch.matmul(p_cycle_coords_centered.permute(0, 1, 3, 2), q_cycle_coords_centered.unsqueeze(0))
    u, s, v = torch.svd(H)
    d = torch.sign(torch.det(torch.matmul(v, u.permute(0, 1, 3, 2))))
    R_1 = torch.diag_embed(torch.ones([p_cycle_coords.size(0), q_cycle_coords.size(0), 3]))
    R_1[:, :, 2, 2] = d
    R = torch.matmul(v, torch.matmul(R_1, u.permute(0, 1, 3, 2)))
    b = q_cycle_coords[:, q_mask].mean(dim=1) - torch.matmul(R, p_cycle_coords[:, :, p_mask].mean(dim=2).unsqueeze(
        -1)).squeeze(-1)

    p_cycle_coords_aligned = torch.matmul(R, p_cycle_coords.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) + b.unsqueeze(2)

    return p_cycle_coords_aligned
