from platform import system

import torch

SYSTEM = system()

def get_torch_device():
    # print(SYSTEM)
    match SYSTEM:
        case "Darwin":
            return "mps" if torch.backends.mps.is_available() else "cpu"
        case "Windows":
            return "cuda" if torch.cuda.is_available() else "cpu"
        case "Linux":
            return "cuda" if torch.cuda.is_available() else "cpu"
        case _:
            return "cpu"

