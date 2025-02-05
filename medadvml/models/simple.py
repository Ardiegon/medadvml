import torch
import torch.nn as nn

MODEL_SIZE_FACTOR = 32

class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear1 = nn.Linear(1, 32*MODEL_SIZE_FACTOR)
        self.linear2 = nn.Linear(32*MODEL_SIZE_FACTOR, 64*MODEL_SIZE_FACTOR)
        self.linear3 = nn.Linear(64*MODEL_SIZE_FACTOR, 64*MODEL_SIZE_FACTOR)
        self.output = nn.Linear(64*MODEL_SIZE_FACTOR, 1)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = (self.output(x))
        return x

