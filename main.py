import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD,Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
from convnext import convnext_small


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
class KMNISTDataset(Dataset):
    def __init__(self, fname_list, label_list, image_dir, transform=None):
        super().__init__()
        self.fname_list = fname_list
        self.label_list = label_list
        self.image_dir = image_dir
        self.transform = transform
        
    # Datasetを実装するときにはtorch.utils.data.Datasetを継承する
    # __len__と__getitem__を実装する
    
    def __len__(self):
        return len(self.fname_list)
    
    def __getitem__(self, idx):
        fname = self.fname_list[idx]
        label = self.label_list[idx]
        
        image = cv2.imread(os.path.join(self.image_dir, fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.transform is not None:
            image = self.transform(image)
        # __getitem__でデータを返す前にtransformでデータに前処理をしてから返すことがポイント
        return image, label

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels=12, kernel_size=3)
        
        self.drop = nn.Dropout(0.25)
        self.batch1 = nn.BatchNorm2d(6)
        self.batch2 = nn.BatchNorm2d(12)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=12*5*5,out_features=10)

    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.maxpool(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

def accuracy_score_torch(y_pred, y):
    y_pred = torch.argmax(y_pred, axis=1).cpu().numpy()
    y = y.cpu().numpy()

    return accuracy_score(y_pred, y)

if __name__ == "__main__":
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
        'batch_size': 64,
        'epochs': 30,
        'lr': 0.001,
        'valid_batch_size': 256,
        'test_batch_size': 256,
    }


    state_dicts = []
    best_sd = None
    best_score = 0

    wandb.init = wandb.init(project='labcompe', entity='keiomobile2', config=PARAMS)
    
    
    train_df = pd.read_csv(PATH['train'])
    sample_submission_df = pd.read_csv(PATH['sample_submission'])
    
    # net = Net()
    train_df, valid_df = train_test_split(
    train_df, test_size=PARAMS['valid_size'], random_state=SEED, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.RandomRotation(10), #id1

    #     transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)), # id4

    #     # numpy.arrayで読み込まれた画像をPyTorch用のTensorに変換します．
    #     transforms.Normalize((0.5, ), (0.5, ))
    #     #正規化の処理も加えます。
    # ])

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
    


    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[ID], train_df[TARGET])):
        # Setting of model
        # model = ResNet().to(DEVICE)
        # model = convnext_tiny(pretrained=True).to(DEVICE)
        model = convnext_small(pretrained=True).to(DEVICE)

        optim = Adam(model.parameters(), lr=PARAMS['lr'])
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

    test_dataset = KMNISTDataset(
        sample_submission_df[ID],
        sample_submission_df[TARGET],
        PATH['test_image_dir'],
        transform=transform
    )

    test_dataloader = DataLoader(test_dataset, batch_size=PARAMS['test_batch_size'], shuffle=False)
    
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
