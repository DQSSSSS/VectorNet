import torch

r"""
Reference:
https://eval.ai/web/challenges/challenge-page/454/evaluation
https://github.com/argoai/argoverse-api/blob/master/argoverse/evaluation/eval_forecasting.py
"""

def get_ADE(a, b):
    r"""
    Calculate Average Displacement Error(ADE).
    Args:
        a: [batch_size, len, dim]
        b:
    Returns: 
        ADE, \frac{1}{n} \sum sqrt((A_x_i-B_x_i)^2 + (A_y_i-B_y_i)^2)
        [batch_size, 1]
    """
    assert a.shape == b.shape
    tmp = torch.sqrt(torch.sum((a - b) ** 2, dim=2)) # [batch_size, len]
    ade = torch.mean(tmp, dim=1, keepdim=True) # [batch_size, 1]
    return ade 

def get_FDE(a, b):
    r"""
    Calculate Final Displacement Error(FDE).
    Args:
        a: [batch_size, len, dim]
        b:
    Returns: 
        FDE, [batch_size, 1]
    """
    assert a.shape == b.shape
    a = a[:, -1, :] # [batch_size, dim]
    b = b[:, -1, :] 
    fde = torch.sqrt(torch.sum((a - b) ** 2, dim=1, keepdim=True)) # [batch_size, 1]
    return fde 

def get_DE(a, b, t_list):
    r"""
    Calculate Displacement Error(DE) at time `t` in `t_list`.
    Args:
        a: [batch_size, len, dim]
        b:
        t_list (list): len(t_list)=n
    Returns: 
        DE, [batch_size, n]
    """
    t_tensor = torch.tensor(t_list)
    a = torch.index_select(a, 1, t_tensor) # [batch_size, n, dim]
    b = torch.index_select(b, 1, t_tensor)
    de = torch.sqrt(torch.sum((a - b) ** 2, dim=2))
    return de