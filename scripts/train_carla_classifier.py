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
    model = 'efficientnet_b7'
    with_transforms = True


# --------- helpers ----------


def last_layer(in_dim, out_dim):
    return nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                         nn.Linear(in_dim, out_dim))


def get_cell(target):
    count, _, _ = np.histogram2d([target[0]], [target[1]],
                                 bins=10,
                                 range=[[-10, 400], [-10, 400]])
    cell = count.flatten().nonzero()[0]
    return cell


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

        cell = get_cell(target)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, cell

    def __len__(self):
        return len(self.imgs)


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

    if args.model == 'efficientnet_b7':
        model_conv = torchvision.models.efficientnet_b7(pretrained=True)
        model_conv.classifier = last_layer(2560, 100)
    elif args.model == 'resnet152':
        model_conv = torchvision.models.resnet152(pretrained=True)
        in_features = model_conv.fc.in_features
        model_conv.fc = nn.Linear(in_features, 100)

    args.model = model_conv.to(args.device)

    return args


def train(args):
    best_model_wts = copy.deepcopy(args.model.state_dict())
    best_loss = np.float('inf')

    model = args.model
    dataloaders = args.dataloaders
    device = args.device

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=7,
        gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs

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
            running_acc = 0.0

            # Iterate over data.
            for inputs, cells in dataloaders[phase]:
                inputs = inputs.to(device)
                cells = cells.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(inputs)
                    loss = criterion(pred, cells.flatten())
                    _, y_pred = torch.max(pred, 1)
                    acc = torch.mean((y_pred == cells.flatten()).float())
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_acc += acc.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / args.dataset_sizes[phase]
            epoch_acc = running_acc / args.dataset_sizes[phase]

            losses[f'{phase}_loss'] = epoch_loss
            losses[f'{phase}_acc'] = epoch_acc

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
    wandb_run = wandb.init(
        project=WANDB_PROJECT_NAME,
        config=_config,
        tags=[_run.experiment_info['name']],
        settings=wandb.Settings(start_method='fork'),
    )
    args = init(_config)
    model = train(args)

    # "model.pth" is saved in wandb.run.dir & will be uploaded at the end of training
    torch.save(model.state_dict(), os.path.join(wandb.run.dir,
                                                'model_best.pth'))
