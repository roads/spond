
import torch
import torch.distributions as D

"""Losses module.
​
Functions:
    cycle_loss: Calculate cycle consistency loss for a system and its 
    mapping back to itself through a model (l1 norm of distances 
    between points)

    create_gmm: Generate probability distribution using gaussian 
    kernels on a system of points

    negloglik: Calculate loglikelihood of drawing a sample from a 
    probability distribution

    mapping_accuracy: Accuracy of a mapping vs ground truth

    pairwise_distance: Calculate pairwise distances between points in 
    two systems

    np_alignment_corr: Calculate alignment correlation between two 
    systems with mapping in both directions (A->B and B->A)

"""

def _euclidean_distance(x, y):
    """L2 distance."""
    return torch.mean(torch.sum(torch.pow(x - y, 2), 1))


def sup_loss_func(x_sup, f_y_sup, loss_sup_scale):
    """Supervised loss."""
    loss = loss_sup_scale * _euclidean_distance(x_sup, f_y_sup)
    return loss


# FLAG DIFFERENCES FROM KENGO VERSION: mine allows for cycle loss in one system only
def cycle_loss_flex(X, gf_x, Y=None, fg_y=None, loss_cycle_scale=1):
    """
    Calculate cycle consistency loss for a system and its mapping back
    to itself through the model (l1 norm of distances between points)

    Args:
     - X: Original system, tensor
     - gf_x: Resulting system for comparison to original. Tensor with 
     same shape as X. Assumes points correspond to those in X
     - Y and gf_y (optional): Second system

    Output:
     - tot_loss: cycle loss per concept
    """

    if Y == None:
        loss = loss_cycle_scale * _euclidean_distance(X, gf_x)

    elif Y is not None and fg_y is not None:

        loss = (loss_cycle_scale * 0.5 
                * _euclidean_distance(X, gf_x) + _euclidean_distance(Y, fg_y)
                )

    return loss


# FLAG DIFFERENCES FROM KENGO VERSION: have separate gmm formation function and nll function
def create_gmm(system, gmm_scale=0.05):

    """
    Generate probability distribution using gaussian kernels on a
    system of points

    Args:
     - system: set of points from which gmm will be produced
     - batches: bool indicating if system shape includes batch dimension
     - kernel_size: stdev of kernel placed on each point to form gmm

    Output: 
     - gmm_x: gmm probability distribution
    """

    system = torch.squeeze(system)
    n_dim = system.shape[-1]
    n_concepts = system.shape[-2]

    # Weight concepts equally
    mix = D.Categorical(torch.ones(n_concepts,))
    
    # Covariance matrix (diagonal) set with gmm_scale
    components = D.Independent(D.Normal(system, gmm_scale * torch.ones(n_dim,)), 1)
    gmm_X = D.mixture_same_family.MixtureSameFamily(mix, components)
    
    return gmm_X


def negloglik(dist, sample, dist_loss_scale):
    """
    Calculate loglikelihood of drawing a sample from a probability
    distribution

    Args:
     - dist: probability distribution (e.g, output of create_gmm)
    """
    result = -torch.mean(dist.log_prob(sample.double()), axis = 0)
    return result
