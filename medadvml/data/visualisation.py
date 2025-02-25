import numpy as np
import matplotlib.pyplot as plt

# def imshow(inp, title=None):
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated

def imshow(img, ax=None):
    img = img.permute(1, 2, 0).numpy()
    img = img * 0.225 + 0.45
    img = img.clip(0, 1)
    if ax is None:
        plt.imshow(img)
        plt.axis('off')
    else:
        ax.imshow(img)
        ax.axis('off')