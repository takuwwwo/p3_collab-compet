import torch


def hard_update(target: torch.nn.Module, source: torch.nn.Module):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
    for target_buffer, buffer in zip(target.buffers(), source.buffers()):
        target_buffer.data.copy_(buffer.data)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    for target_buffer, buffer in zip(target.buffers(), source.buffers()):
        target_buffer.data.copy_(target_buffer.data * (1.0 - tau) + buffer.data * tau)
