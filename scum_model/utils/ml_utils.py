import torch 

def compute_grad_stats(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:  # Check if the gradient is computed
            grads.append(param.grad.view(-1))  # Flatten the gradient tensor

    if grads:
        all_grads = torch.cat(grads)  # Concatenate all gradients into one tensor
        median_grad = all_grads.median()
        mean_grad = all_grads.mean()
        max_grad = all_grads.max()
        min_grad = all_grads.min()
        p99_grad = torch.quantile(all_grads, 0.99)
        p01_grad = torch.quantile(all_grads, 0.01)
        return median_grad, mean_grad, max_grad, min_grad, p99_grad, p01_grad
    else:
        return None, None  # No gradients present (e.g., in untrained parameters)