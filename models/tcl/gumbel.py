import torch


def gumbel_sigmoid(logits: torch.Tensor, tau:float = 1, hard: bool = False):

    logistic = torch.rand_like(logits)
    logistic = logistic.div_(1. - logistic).log_()  # ~Logistic(0,1)

    gumbels = (logits + logistic) / tau  # ~Logistic(logits, tau)
    y_soft = gumbels.sigmoid_()

    if hard:
        # Straight through.
        y_hard = y_soft.gt(0.5).type(y_soft.dtype)
        # gt_ break gradient flow
        #  y_hard = y_soft.gt_(0.5)  # gt_() maintain dtype, different to gt()
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    return ret
