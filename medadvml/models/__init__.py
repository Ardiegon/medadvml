import os
import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from tempfile import TemporaryDirectory

from medadvml.config import Config
from medadvml.models.simple import MedModel
from medadvml.utilities import platform
from medadvml.data.visualisation import imshow 


class ModelWrapper():
    def __init__(self, config):
        num_labels = len(config["label"].keys())
        self.name = config.name 
        self.task = config["task"]
        self.model = MedModel(num_labels=num_labels)
        self.device = platform.get_torch_device()
        self.model.to(self.device)

        self.model_path = os.path.join("medadvml", "weights", f'{self.name}_params.pt')

        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
            print("Weights Loaded!")

        
    def fit(self, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        self.model_path = os.path.join("medadvml", "weights", f'{self.name}_params.pt')

        torch.save(self.model.state_dict(), self.model_path)
        best_acc = 0.0
        

        for epoch in tqdm(range(num_epochs), desc="Epochs: "):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train() 
                else:
                    self.model.eval() 

                running_loss = 0.0
                running_corrects = 0

                pbar = tqdm(dataloaders[phase], desc="Inference - 0.0 ", leave=False)
                for inputs, labels in pbar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            if self.task == 'multi-label, binary-class':
                                labels = labels.to(torch.float32)
                                loss = criterion(outputs, labels)
                            else:
                                labels = labels.squeeze().long()
                                loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                    pbar.set_description(f"{phase} - {loss.item():.5f}")
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(self.model.state_dict(), self.model_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))

    # def visualize(self, dataloaders, class_names, num_images=6):
    #     was_training = self.model.training
    #     self.model.eval()
    #     images_so_far = 0
    #     fig = plt.figure()

    #     with torch.no_grad():
    #         for i, (inputs, labels) in enumerate(dataloaders['val']):
    #             inputs = inputs.to(self.device)
    #             labels = labels.to(self.device)

    #             outputs = self.model(inputs)
    #             _, preds = torch.max(outputs, 1)

    #             for j in range(inputs.size()[0]):
    #                 images_so_far += 1
    #                 ax = plt.subplot(num_images//2, 2, images_so_far)
    #                 ax.axis('off')
    #                 ax.set_title(f'predicted: {class_names[preds[j]]}')
    #                 imshow(inputs.cpu().data[j])


    #                 if images_so_far == num_images:
    #                     self.model.train(mode=was_training)
    #                     return
    #         self.model.train(mode=was_training)

    def visualize(self, dataloaders, class_names, num_images=16):
        self.model.eval()
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()

        images_shown = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                for i in range(inputs.size(0)):
                    if images_shown >= num_images:
                        break
                    ax = axes[images_shown]
                    imshow(inputs[i].cpu(), ax=ax)
                    ax.set_title(f'True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}', fontsize=10)
                    ax.axis('off')
                    images_shown += 1

                if images_shown >= num_images:
                    break

        plt.tight_layout()
        plt.show()

    def predict(self, x):
        return self.model(x)
