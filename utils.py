import torch

def save_model(model, path):
    """
    Saves the model state dictionary to the specified path.
    """
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    Loads the model state dictionary from the specified path.
    """
    model.load_state_dict(torch.load(path, weights_only=True))
