import torch
import timm

def build_model(name: str, pretrained: bool = True):
    """Create a pretrained backbone with a fresh one-logit head."""
    return timm.create_model(name, pretrained = pretrained, num_classes = 1)


def get_device() -> torch.device:
    """Pick the best available device: CUDA -> Apple MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")