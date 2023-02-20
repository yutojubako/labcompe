import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD,Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from models.mynet import Mynet
from sklearn.model_selection import StratifiedKFold
from models.convnext import convnext_tiny
from utils.utils import seed_everything, accuracy_score_torch
from datasets.dataset import KMNISTDataset
from models.resnet import ResNet
import argparse

def parse_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--folds', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model', type=str, required=True,choices=['resnet', 'convnext', 'mynet'])
    parser.add_argument('--wandb', type=bool, required=True,choices=[True, False])
    parser.add_argument('--optim', type=str, default='adam',choices=['adam', 'sgd'])
    args = parser.parse_args()
    
    return args






def main():
    args = parse_params()
    
    INPUT_DIR = './input/bootcamp2023/'

    PATH = {
        'train': os.path.join(INPUT_DIR, 'train.csv'),
        'sample_submission': os.path.join(INPUT_DIR, 'sample_submission.csv'),
        'train_image_dir': os.path.join(INPUT_DIR, 'train_images/train_images'),
        'test_image_dir': os.path.join(INPUT_DIR, 'test_images/test_images'),
    }

    ID = 'fname'
    TARGET = 'label'

    SEED = 42
    seed_everything(SEED)

    # GPU settings for PyTorch (explained later...)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parameters for neural network. We will see the details later...
    PARAMS = {
        'valid_size': 0.2,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': 0.001,
        'valid_batch_size': 256,
        'test_batch_size': 256,
    }


    state_dicts = []
    best_sd = None
    best_score = 0

    if args.wandb:
        wandb.init = wandb.init(project='labcompe', entity='keiomobile2', config=PARAMS)
    
    
    train_df = pd.read_csv(PATH['train'])
    sample_submission_df = pd.read_csv(PATH['sample_submission'])
    
    # net = Net()
    train_df, valid_df = train_test_split(
    train_df, test_size=PARAMS['valid_size'], random_state=SEED, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomOrder([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.1), ratio=(0.9, 1.1)),
            transforms.RandomAffine(degrees=10)
        ]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    ])

    train_dataset = KMNISTDataset(train_df[ID], train_df[TARGET], PATH['train_image_dir'], transform=transform)
    valid_dataset = KMNISTDataset(valid_df[ID], valid_df[TARGET], PATH['train_image_dir'], transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=PARAMS['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=PARAMS['valid_batch_size'], shuffle=False)
    


    k = args.folds
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[ID], train_df[TARGET])):
        # Setting of model
        if args.model == 'resnet':
            model = ResNet().to(DEVICE)
        elif args.model == 'convnext':
            model = convnext_tiny(pretrained=True).to(DEVICE)
        elif args.model == 'mynet':
            model = Mynet().to(DEVICE)

        
        if args.optim == 'adam':
            optim = Adam(model.parameters(), lr=PARAMS['lr'])  
        if args.optim == 'sgd':
            optim = SGD(model.parameters(), lr=PARAMS['lr'], momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.8, patience=1)

        # Setting of dataset
        tr_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_df.iloc[valid_idx].reset_index(drop=True)

        train_dataset = KMNISTDataset(tr_df[ID], tr_df[TARGET], PATH['train_image_dir'], transform=transform)
        valid_dataset = KMNISTDataset(val_df[ID], val_df[TARGET], PATH['train_image_dir'], transform=transform)

        train_dataloader = DataLoader(train_dataset, batch_size=PARAMS['batch_size'], shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=PARAMS['valid_batch_size'], shuffle=False)


        # Training
        for epoch in range(PARAMS['epochs']):
            # epochループを回す
            model.train()
            train_loss_list = []
            train_accuracy_list = []
            
            for x, y in tqdm(train_dataloader):
                # 先ほど定義したdataloaderから画像とラベルのセットのdataを取得
                x = x.to(dtype=torch.float32, device=DEVICE)
                y = y.to(dtype=torch.long, device=DEVICE)
                
                # pytorchでは通常誤差逆伝播を行う前に毎回勾配をゼロにする必要がある
                optim.zero_grad()
                # 順伝播を行う
                y_pred = model(x)
                # lossの定義 今回はcross entropyを用います
                loss = criterion(y_pred, y)
                # 誤差逆伝播を行なってモデルを修正します(誤差逆伝播についてはhttp://hokuts.com/2016/05/29/bp1/)
                loss.backward() # 逆伝播の計算
                # 逆伝播の結果からモデルを更新
                optim.step()
                
                train_loss_list.append(loss.item())
                train_accuracy_list.append(accuracy_score_torch(y_pred, y))
                if args.wandb:
                    wandb.log({'train_loss': loss.item(), 'train_accuracy': accuracy_score_torch(y_pred, y)})

            model.eval()
            valid_loss_list = []
            valid_accuracy_list = []

            for x, y in tqdm(valid_dataloader):
                x = x.to(dtype=torch.float32, device=DEVICE)
                y = y.to(dtype=torch.long, device=DEVICE)
                
                with torch.no_grad():
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                
                valid_loss_list.append(loss.item())
                valid_accuracy_list.append(accuracy_score_torch(y_pred, y))
                if args.wandb:
                    wandb.log({'valid_loss': loss.item(), 'valid_accuracy': accuracy_score_torch(y_pred, y)})
            
            print('epoch: {}/{} - loss: {:.5f} - accuracy: {:.3f} - val_loss: {:.5f} - val_accuracy: {:.3f}'.format(
                epoch,
                PARAMS['epochs'], 
                np.mean(train_loss_list),
                np.mean(train_accuracy_list),
                np.mean(valid_loss_list),
                np.mean(valid_accuracy_list)
            ))
        
            if best_score <= np.mean(valid_accuracy_list):
                best_score = np.mean(valid_accuracy_list)
                best_sd = model.state_dict()
            scheduler.step(np.mean(valid_loss_list))
    state_dicts.append(best_sd)
    
    torch.save(best_sd, f'./best_{args.model}_fold{fold}.pth')

    test_dataset = KMNISTDataset(
        sample_submission_df[ID],
        sample_submission_df[TARGET],
        PATH['test_image_dir'],
        transform=transform
    )

    test_dataloader = DataLoader(test_dataset, batch_size=PARAMS['test_batch_size'], shuffle=False)
    
    model.load_state_dict(torch.load(f'./best_{args.model}_fold{fold}.pth'))
    model.eval()
    predictions = []

    for x, _ in test_dataloader:
        x = x.to(dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, axis=1).cpu().numpy()
            y_pred = y_pred.tolist()
            
        predictions += y_pred
        
    sample_submission_df[TARGET] = predictions
    
    sample_submission_df.to_csv('submission.csv', index=False)
    from IPython.display import FileLink
    FileLink('submission.csv')


if __name__ == '__main__':
    main()