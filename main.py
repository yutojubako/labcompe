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

# 入力は28*28の白黒画像で10クラス分類を行う

class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=128, output_dim=10):
        super().__init__()
        # nn.Linearは fully-connected layer (全結合層)のことです．
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # 1次元のベクトルにする
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        return x

# 以下を埋めてみよう
# 今回の研修では
# モデルとして入力から出力チャネル数6, kernel_size5の畳み込み層→Maxpooling(2×2)→出力チャネル数12, kernel_size3の畳み込み層
# → MaxPooling(2×2)→1次元にする→Linearで10次元出力
# というモデルを作成してください(strideなどは考えないでください)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 出力チャンネル数6, kernel size 5のCNNを定義する
        # 畳み込みの定義はPytorchの場合torch.nn.Conv2dで行います。ヒント:白黒画像とはチャネル数いくつかは自分で考えよう
        # 公式documentで使い方を確認する力をつけてほしいので、自分でconv2dなどの使い方は調べよう
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=3)
        # 出力チャネル数12, kernel_size 3のCNNを定義する 上記と同様に今度は自分で書いてみよう
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=5)
        
        self.drop = nn.Dropout(0.25)

        # Maxpoolingの定義(fowardでするのでもどっちでも)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Linearの定義
        # 線形変換を行う層を定義してあげます: y = Wx + b
        # self.conv1, conv2のあと，maxpoolingを通すことで，
        # self.fc1に入力されるTensorの次元は何になっているか計算してみよう！
        # これを10クラス分類なので，10次元に変換するようなLinear層を定義します
        
        self.fc1 = nn.Linear(in_features=64*5*5,out_features=10)

    
    def forward(self, x):
        batch_size = x.shape[0]
        # forward関数の中では，，入力 x を順番にレイヤーに通していきます．みていきましょう．    
        # まずは，画像をCNNに通します
        x = self.conv1(x)

        # 活性化関数としてreluを使います
        x = F.relu(x)
        
        # 次に，MaxPoolingをかけます．
        x = self.maxpool(x)
        
        # 2つ目のConv層に通します
        x = self.conv2(x)
        
        # MaxPoolingをかけます
        x = self.maxpool(x)
        
        x = self.drop(x)

        # 少しトリッキーなことが起きます．
        # CNNの出力結果を fully-connected layer に入力するために
        # 1次元のベクトルにしてやる必要があります
        # 正確には，　(batch_size, channel, height, width) --> (batch_size, channel * height * width)
        x = x.view(batch_size, -1)
        
        # linearと活性化関数に通します
        x = self.fc1(x)
#         x = F.relu(x)
        return x
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
        'epochs': 25,
        'lr': 0.001,
        'valid_batch_size': 256,
        'test_batch_size': 256,
    }


    
    wandb.init = wandb.init(project='labcompe', entity='keiomobile2', config=PARAMS)
    
    
    train_df = pd.read_csv(PATH['train'])
    sample_submission_df = pd.read_csv(PATH['sample_submission'])
    
    net = Net()
    train_df, valid_df = train_test_split(
    train_df, test_size=PARAMS['valid_size'], random_state=SEED, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomRotation(10), #id1
        # numpy.arrayで読み込まれた画像をPyTorch用のTensorに変換します．
        transforms.Normalize((0.5, ), (0.5, ))
        #正規化の処理も加えます。
    ])

    train_dataset = KMNISTDataset(train_df[ID], train_df[TARGET], PATH['train_image_dir'], transform=transform)
    valid_dataset = KMNISTDataset(valid_df[ID], valid_df[TARGET], PATH['train_image_dir'], transform=transform)

    # DataLoaderを用いてバッチサイズ分のデータを生成します。shuffleをtrueにすることでデータをshuffleしてくれます
    train_dataloader = DataLoader(train_dataset, batch_size=PARAMS['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=PARAMS['valid_batch_size'], shuffle=False)
    
    model = MLP().to(DEVICE)
    
    optim = Adam(model.parameters(), lr=PARAMS['lr'])
    criterion = nn.CrossEntropyLoss()
    
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
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

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
