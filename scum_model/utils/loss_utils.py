import torch

def compute_policy_error(policy_log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    policy_loss = -policy_log_probs  * advantages.unsqueeze(1).detach()
    return policy_loss.mean()


def compute_policy_error_with_clipped_surrogate(policy_log_probs: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor, clip_range: float=.2) -> torch.Tensor:
    ratio = torch.exp(policy_log_probs - old_log_probs)
    # clipped surrogate loss
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = - torch.min(policy_loss_1, policy_loss_2)
    return policy_loss, ratio


def compute_entropy(policy_probs: torch.Tensor, policy_log_probs: torch.Tensor) -> torch.Tensor:
    return -(policy_probs * policy_log_probs.unsqueeze(1)).sum(dim=1).mean()
