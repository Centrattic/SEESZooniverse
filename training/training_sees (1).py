import os
import shutil

import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from PIL import Image
import cv2

from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, Checkpoint, EpochScoring, EarlyStopping
from skorch.dataset import Dataset
from skorch.helper import predefined_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from pathlib import Path
import tensorflow as tf
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")

# model definitions
class PretrainedModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model

    def forward(self, x):
        return self.model(x)
    
# train code

def train(data_dir, experiment_name,
          num_classes=2, batch_size=64, num_epochs=50, lr=0.00005, image_size = (224, 224)): 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if gpu available
    if device == 'cuda': # using all available gpus
        torch.cuda.empty_cache()

    f_params = f'./outputs/checkpoints/{experiment_name}/model.py'
    f_history = f'./outputs/histories/{experiment_name}/model.json'
    csv_name = f'./outputs/probabilities/{experiment_name}/model.csv'
        
        
    # transforms to data for more variability 
    train_transforms = transforms.Compose([transforms.Resize(image_size),
					   transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomRotation(100),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(image_size),
					  transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # train folders
    train_folder = os.path.join(data_dir, 'train')
    val_folder = os.path.join(data_dir, 'val')
    test_folder = os.path.join(data_dir, 'test')

    train_dataset = datasets.ImageFolder(train_folder, train_transforms)
    val_dataset = datasets.ImageFolder(val_folder, test_transforms)
    test_dataset = datasets.ImageFolder(test_folder, test_transforms)
    
    print ("Train/Test/Val datasets have been created.")

    labels = np.array(train_dataset.samples)[:,1]
    
    # weighting the classes properly
    labels = labels.astype(int)
    class0_weight = 1 / len(labels[labels == 0]) # vs 1 - len(labels[labels == 0])/len(labels) see if this is better or WAY too much
    class1_weight = 1 / len(labels[labels == 1])
    sample_weights = np.array([class0_weight, class1_weight])
    weights = sample_weights[labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_dataset), replacement=True)

    print()
    print(f'Experiment Name:{experiment_name}')
    print(f'Data Directory: {data_dir}')
    print(f'Number of Classes: {num_classes}')
    print(f'Number of class 0: {len(labels[labels == 0])}')
    print(f'Number of class 1: {len(labels[labels == 1])}')
    print(f'Batch Size: {batch_size}')
    print(f'Number of Epochs: {num_epochs}')
    print(f'Initial Learning Rate: {lr}')
    print(f'Device: {device}')
    print(f'quick test3')
    print()

    # model checkpointing to save
    
    checkpoint = Checkpoint(monitor='valid_loss_best',
                            f_params=f_params,
                            f_history=f_history,
                            f_optimizer=None,
                            f_criterion=None)

    # model training reporting
    
    train_acc = EpochScoring(scoring='accuracy',
                             on_train=True,
                             name='train_acc',
                             lower_is_better=False)

    # model early stopping 

    early_stopping = EarlyStopping(patience=8)

    callbacks = [checkpoint, train_acc, early_stopping]
    
    cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor(sample_weights))

    # model parameters
    net = NeuralNetClassifier(PretrainedModel,
                              criterion=nn.CrossEntropyLoss(),
                              lr=lr,
                              batch_size=batch_size,
                              max_epochs=num_epochs,
                              module__output_features=num_classes,
                              optimizer=optim.SGD,
                              optimizer__momentum=0.9,
                              iterator_train__num_workers=1,
                              iterator_train__sampler=sampler,
                              iterator_valid__shuffle=False,
                              iterator_valid__num_workers=1,
                              train_split=predefined_split(val_dataset),
                              callbacks=callbacks,
                              device=device)

    print ("Model is fitting. Thank you for your patience.")
    net.fit(train_dataset, y=None)

    print ("Model is performing inference. Results saved in probabilities folder.")

    img_locs = [loc for loc, _ in test_dataset.samples]
    test_probs = net.predict_proba(test_dataset)
    test_probs = [prob[0] for prob in test_probs] # probability of being black
    data = {'img_loc' : img_locs, 'probability' : test_probs}
    pd.DataFrame(data=data).to_csv(csv_name, index=False)
    
    print ("The code is done.")

    
# running the training

if __name__ == '__main__':
    
    experiment_name = "water"
    data_dir = "datasets/water"
    
    if not os.path.isdir(os.path.join('outputs', 'probabilities', experiment_name)):
        os.makedirs(os.path.join('outputs', 'probabilities', experiment_name))
    if not os.path.isdir(os.path.join('outputs', 'checkpoints', experiment_name)):
        os.makedirs(os.path.join('outputs', 'checkpoints', experiment_name))
    if not os.path.isdir(os.path.join('outputs', 'histories', experiment_name)):
        os.makedirs(os.path.join('outputs', 'histories', experiment_name))

    train(data_dir, experiment_name)