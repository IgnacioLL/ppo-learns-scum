import torch
import pandas as pd
import os
import uuid
import random
from torch.utils.tensorboard import SummaryWriter

def initialize_metrics():
    return {
        'value_loss': 0,
        'policy_loss': 0,
        'entropy': 0,
        'total_loss': 0,
        'value_prediction_avg': 0,
        'returns': 0,
        'ratio': 0,
        'mean_gradient': 0,
        'median_gradient': 0,
        'max_gradient': 0,
        'min_gradient': 0,
        'p99_gradient': 0,
        'p01_gradient': 0,
        'advantadge': 0,
        'advantadge_normalized': 0,
        'learning_rate': 0,
        "variance_in_logits": 0,
        'mean_change_ratio': 0, 
        'max_change_ratio': 0, 
        'std_change_ratio': 0, 
        'max_prob': 0
    }


def accumulate_metrics(total_metrics, batch_metrics):
    for key in total_metrics:
        total_metrics[key] += batch_metrics[key]


def average_metrics(metrics, total_samples, batch_size):
    num_batches = (total_samples + batch_size - 1) // batch_size
    return {key: value / num_batches for key, value in metrics.items()}


def get_gradient_stats(model):
    median_grad, mean_grad, max_grad, min_grad, p99_grad, p01_grad = compute_grad_stats(model)
    return {
        'mean_gradient': mean_grad.item(),
        'median_gradient': median_grad.item(),
        'max_gradient': max_grad.item(),
        'min_gradient': min_grad.item(),
        'p99_gradient': p99_grad.item(),
        'p01_gradient': p01_grad.item(),
    }

def compute_grad_stats(model: torch.nn.Module):
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
    
def log_current_state_and_prediction(state, prediction):
    if int(random.random()*100_000) % 50_000 == 0:
        print("State is: ", state)
        print("Prediction: ", prediction[0])

def log_data_being_trained(*args: torch.Tensor, episode=""):
    epoch = args[-1]
    columns = ["value_preds", "batch_returns", "batch_action", "batch_log_prob", "log_prob", 
            "advantadge", "advantadge_norm", "policy_loss", "ratio", "value_loss", "epoch"]
    
    # Create a copy of tensors for logging without affecting the computational graph
    tensor_copies = []
    for tensor in args[:-1]:  # Exclude epoch
        # Clone the tensor and detach it in a separate operation
        with torch.no_grad():
            tensor_copy = tensor.clone().cpu().detach().numpy()
        tensor_copies.append(tensor_copy)
    
    df = pd.DataFrame(tensor_copies, index=columns[:-1]).T
    df['epoch'] = epoch
    
    # Ensure directory exists
    os.makedirs("./analytics/data/training_data/", exist_ok=True)
    
    # Save with unique identifier
    uuid_ = uuid.uuid4()
    df.to_parquet(f"./analytics/data/training_data/training_data_{episode}_{uuid_}_{epoch}.parquet")


def flush_performance_stats_tensorboard(writer: SummaryWriter, performance_stats: dict, episode: int) -> None:
    for stat in performance_stats:  
        writer.add_scalar(f"{stat}", performance_stats[stat], episode)
    writer.flush()

def flush_average_reward_to_tensorboard(writer: SummaryWriter, average_reward: float, episode: int):
    average_rewards_format = {"Avg Reward": average_reward}
    flush_performance_stats_tensorboard(writer, average_rewards_format, episode)

def flush_average_win_rate_to_tensorboard(writer: SummaryWriter, win_rate: float, episode: int):
    win_rate_formatted = {"Win Rate": win_rate}
    flush_performance_stats_tensorboard(writer, win_rate_formatted, episode)
