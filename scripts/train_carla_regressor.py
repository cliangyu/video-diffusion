# Script adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
import copy
import os
import sys
import time
from pathlib import Path
from time import sleep

import ml_helpers as mlh
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from sacred import Experiment
from torch.optim import lr_scheduler
from torchvision import transforms

import wandb

# Use sacred for command line interface + hyperparams
# Use wandb for experiment tracking

ex = Experiment()
WANDB_PROJECT_NAME = 'carla-regressor'
if '--unobserved' in sys.argv:
    os.environ['WANDB_MODE'] = 'dryrun'

# Put all hyperparameters + paths in my_config().
# and more complex data objects in init()


@ex.config
def my_config():
    # paths
    home_dir = '.'  # required by job submitter, don't modify
    artifact_dir = './artifacts/'  # required by job submitter, don't modify

    # relative to home_dir
    data_dir = '/ubc/cs/research/plai-scratch/video-diffusion-shared/datasets/carla-no-traffic-regression/'

    # Hyper params
    lr = 0.001
    batch_size = 4

    # Training settings
    num_epochs = 25
    seed = 0
    with_classifier = False
    model = 'resnet18'
    with_transforms = True


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def init(config):
    args = mlh.default_init(config)
    args.data_dir = Path(args.data_dir)

    # ===============================
    # ------- set up data -----------
    # ===============================
    if args.with_transforms:
        data_transforms = {
            'train':
            transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToPILImage(),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.ColorJitter(brightness=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ]),
            'test':
            transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ]),
        }
    else:
        data_transforms = {
            'train':
            transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                # transforms.ToPILImage(),
                # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                # transforms.ColorJitter(brightness=.1, hue=.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ]),
            'test':
            transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ]),
        }

    args.dataloaders = {
        'train':
        torch.utils.data.DataLoader(
            CARLADataset(args.data_dir / 'train', data_transforms['train']),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
        ),
        'test':
        torch.utils.data.DataLoader(
            CARLADataset(args.data_dir / 'test', data_transforms['test']),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
        ),
    }

    args.dataset_sizes = {
        x: len(args.dataloaders[x].dataset)
        for x in ['train', 'test']
    }
    args.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')

    # ===============================
    # ------- set up model ----------
    # ===============================

    if args.model == 'resnet18':
        model_conv = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model_conv.fc.in_features
        last_layer = (nn.Linear(num_ftrs, 3)
                      if args.with_classifier else nn.Linear(num_ftrs, 2))
        model_conv.fc = last_layer

    elif args.model == 'efficientnet_b7':
        model_conv = torchvision.models.efficientnet_b7(pretrained=True)
        num_ftrs = model_conv.classifier[1].in_features
        last_layer = (nn.Linear(num_ftrs, 3)
                      if args.with_classifier else nn.Linear(num_ftrs, 2))
        model_conv.classifier[1] = last_layer

    args.model = model_conv.to(args.device)
    args.criterion = nn.MSELoss()
    args.optimizer = optim.SGD(model_conv.parameters(),
                               lr=args.lr,
                               momentum=0.9)
    args.scheduler = lr_scheduler.StepLR(
        args.optimizer, step_size=7,
        gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs

    return args


class CARLADataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = sorted([p.parts[-1] for p in self.root.glob('video*.npy')])
        self.labels = sorted(
            [p.parts[-1] for p in self.root.glob('coords*.npy')])

    def __getitem__(self, idx):
        # load images and masks
        img = np.load(self.root / self.imgs[idx])
        target = np.load(self.root / self.labels[idx])

        # use x,y only
        target = target[[0, 1]]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


def train(args):
    best_model_wts = copy.deepcopy(args.model.state_dict())
    best_loss = np.float('inf')

    model = args.model
    criterion = args.criterion
    optimizer = args.optimizer
    scheduler = args.scheduler
    dataloaders = args.dataloaders
    device = args.device

    # only used if args.with_classifier is True
    classifier_criterion = nn.BCELoss()
    m = nn.Sigmoid()

    metric_logger = mlh.MetricLogger(wandb=wandb)

    for epoch in metric_logger.step(range(args.num_epochs)):
        losses = {}
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if args.with_classifier:
                        reg_loss = criterion(outputs[:, :2], labels[:, :2])
                        clas_loss = classifier_criterion(
                            m(outputs[:, 2]), labels[:, 2])
                        loss = reg_loss + clas_loss
                    else:
                        loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / args.dataset_sizes[phase]

            losses[phase] = epoch_loss

            # deep copy the model
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(
                    model.state_dict(),
                    os.path.join(wandb.run.dir, f'model_{epoch}.pth'),
                )
                wandb.save(os.path.join(wandb.run.dir, f'model_{epoch}.pth'))

        metric_logger.update(**losses)

    print(f'Best val loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


@ex.automain
def command_line_entry(_run, _config):
    wandb_run = wandb.init(project=WANDB_PROJECT_NAME,
                           config=_config,
                           tags=[_run.experiment_info['name']])
    args = init(_config)
    model = train(args)

    # "model.pth" is saved in wandb.run.dir & will be uploaded at the end of training
    torch.save(model.state_dict(), os.path.join(wandb.run.dir,
                                                'model_best.pth'))
