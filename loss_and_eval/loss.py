import torch
from config import device


def loss_func(a, b, alpha=0.1):
    r"""
    Loss in paper. L_traj + `alpha` * L_node
    L_traj:
        Paper's description:
            - L_traj is the negative Gaussian log-likelihood for the groundtruth future trajectories
        The commonly NLLLoss is for classification problem, but this problem doesn't belong to it.
        Using the literal comprehension, -log(Gaussion(a-b)) is equals MSE, so we use MSE loss.
    L_node:
        Relative to node completion, now we set it to zero.
    Args:
        a: [batch_size, len, dim]
        b:
        alpha: blend factor
    Returns:
        A value.
    """
    L_traj = torch.nn.MSELoss()
    L_node = 0
    return L_traj(a, b) + alpha * L_node
