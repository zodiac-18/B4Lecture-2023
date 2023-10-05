#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
B4輪講最終課題 パターン認識に挑戦してみよう
ベースラインスクリプト(Pytorch Lightning版)
特徴量；MFCCの平均（0次項含まず）
識別器；MLP
"""

"""
pytorch-lightning 
    Docs: https://pytorch-lightning.readthedocs.io/
LightningModule
    Docs: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    API Refference: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html
Trainer
    Docs: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    API Refference: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html
"""


import argparse
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torchaudio
from torchinfo import summary
import librosa
import torchaudio.transforms as T
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import Dataset, random_split
import hydra
from omegaconf import DictConfig, OmegaConf

#from ..utils.wn import WN

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../"

def delta(mfcc, l=2):
        """
        Compute ΔMFCC.

        Args:
            mfcc (ndarray): MFCC.
            l (int, optional): Number of frames between which the difference is taken. Defaults to 2.

        Returns:
            ndarray: ΔMFCC.
        """
        mfcc_pad = np.pad(mfcc, [[l, l+1], [0, 0]], "edge")
        k_square = np.sum(np.power(np.arange(-l, l + 1), 2))
        k_sequence = np.arange(-l, l + 1)
        delta_mfcc = np.zeros_like(mfcc)
        for i in range(mfcc.shape[0]):
            delta_mfcc[i] = np.dot(k_sequence, mfcc_pad[i : i + l * 2 + 1])
        delta_mfcc = delta_mfcc / k_square
        return torch.from_numpy(delta_mfcc.astype(np.float32)).clone()

a = 10000

def plot_melspectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    global a
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    a += 1
    plt.savefig(f"fig/{a}.png")

# 線形スペクトログラムを入力とし、入力データからzを出力する
class VITSPosteriorEncoder(pl.LightningModule):
    def __init__(
        self,
        in_x_channels,  # 入力するxのスペクトログラムの周波数の次元
        out_z_channels,  # 出力するzのチャネル数
        hidden_channels,  # 隠れ層のチャネル数
        kernel_size,  # WN内のconv1dのカーネルサイズ
        dilation_rate,  # WN内におけるconv1dのdilationの数値
        n_resblock,  # WN内のResidual Blockの重ねる数
        gin_channels=0,
    ):  # Gated Information Network(GIN)のチャネル数(default=0)
        super(VITSPosteriorEncoder, self).__init__()
        self.in_x_channels = in_x_channels
        self.out_z_channels = out_z_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_resblock
        self.gin_channels = gin_channels

        # 入力スペクトログラムに対して前処理を行う層
        self.preprocess = nn.Conv1d(in_x_channels, hidden_channels, 1)
        # WNを用いて特徴量の抽出を行う
        self.encode = WN(hidden_channels, kernel_size, dilation_rate, n_resblock, gin_channels=gin_channels)
        # ガウス分布の平均と分散を生成する層
        self.projection = nn.Conv1d(hidden_channels, out_z_channels * 2, 1)

    def forward(self, x_spec, x_spec_lengths, g=None):
        # マスクの作成
        # スペクトログラムの3番目の次元(=時間軸)のサイズから最大フレーム数を取得
        max_length = x_spec.size(2)
        # フレームの時間的な位置情報
        progression = torch.arange(max_length, dtype=x_spec_lengths.dtype, device=x_spec_lengths.device)
        # スペクトログラムの各フレームに対してその時間的位置がそのスペクトログラムの長さ未満(=有効)であるかどうかを示すbool値のテンソル
        x_spec_mask = progression.unsqueeze(0) < x_spec_lengths.unsqueeze(1)
        x_spec_mask = torch.unsqueeze(x_spec_mask, 1).to(x_spec.dtype)
        # preprocess層で畳み込みし、maskを適用
        x_spec = self.preprocess(x_spec) * x_spec_mask
        # WNでエンコードして特徴量を抽出
        x_spec = self.encode(x_spec, x_spec_mask, g=g)
        # 特徴量をガウス分布の平均と対数分散に変換する(マスクを適用)
        stats = self.projection(x_spec) * x_spec_mask
        # statsから平均mと対数分散logsを分離する, out_channelsでzのチャネル数を指定
        m, logs = torch.split(stats, self.out_z_channels, dim=1)
        # reparameterization trick(z=μ+εσ)によりzを擬似サンプリング(マスクを適用)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_spec_mask
        return m, logs, z, x_spec_mask
    
class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        #TODO: hard coding
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 10 * 6, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(self.bn1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(self.bn2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(self.bn3(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class train(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
            super().__init__()
            self.model = CNN(input_dim, output_dim)
            print(self.model)
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.train_acc = torchmetrics.Accuracy()
            self.val_acc = torchmetrics.Accuracy()
            self.test_acc = torchmetrics.Accuracy()
            self.confm = torchmetrics.ConfusionMatrix(10, normalize='true')
            
    def create_model(self, input_dim, output_dim, model_n="cnn"):
        """
        MLPモデルの構築
        Args:
            input_dim: 入力の形
            output_dim: 出力次元
        Returns:
            model: 定義済みモデル
        """
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim*50, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, output_dim),
            torch.nn.Softmax(dim=-1)
        )
        # モデル構成の表示
        print(model)
        return model
    
    def forward(self, x):
        return self.model(x.unsqueeze(1))

    def training_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log('train/acc', self.train_acc(pred,y), on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log('val/acc', self.val_acc(pred,y), prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log('test/acc', self.test_acc(pred, y), prog_bar=True, logger=True)
        return {'pred':torch.argmax(pred, dim=-1), 'target':y}
    
    def test_epoch_end(self, outputs) -> None:
        # 混同行列を tensorboard に出力
        preds = torch.cat([tmp['pred'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        confusion_matrix = self.confm(preds, targets)
        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(10), columns=range(10))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='gray_r').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.002)
        return self.optimizer

class FSDD(Dataset):
    def __init__(
        self,
        path_list,
        label, 
        sample_rate,
        fft_size=512,
        hop_size=128,
        win_length=None,
        window=torch.hann_window,
        n_mels=80,
        fmin=0,
        fmax=None,
        aug=False,
        aug_list=[False, False, False, False],
        mixup=True,
        power=2,
        ) -> None:
            super().__init__()
            self.label = label
            self.sample_rate = sample_rate
            self.fft_size = fft_size
            self.hop_size = hop_size
            self.win_length = win_length
            self.window = window
            self.n_mels = n_mels
            self.fmin = fmin
            self.fmax = fmax
            self.power = power
            self.aug = aug
            self.aug_list = aug_list
            self.mixup = mixup
            self.features = self.feature_extraction(path_list)

    def feature_extraction(self, path_list):
        """
        wavファイルのリストから特徴抽出を行いリストで返す
        扱う特徴量はMFCC13次元の平均（0次は含めない）
        Args:
            root: dataset が存在するディレクトリ
            path_list: 特徴抽出するファイルのパスリスト
        Returns:
            features: 特徴量
        """
        datasize = len(path_list)
        #TODO: Hard-coding
        mel_s = True
        if True:
            max_length = 49 * self.hop_size
            features = torch.zeros(datasize, self.n_mels, 50)
            mel_transform = T.MelSpectrogram(
                n_fft=self.fft_size,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                hop_length=self.hop_size,
                win_length=self.win_length,
                window_fn=self.window,
                pad_mode="reflect",
                power=self.power
            )
        else:
            n_mfcc = 20
            features = torch.zeros(datasize, n_mfcc*2)
            transform = torchaudio.transforms.MFCC(n_mfcc=n_mfcc, melkwargs={'n_mels':80, 'n_fft':512})
        
        # Data Augmentation用のリスト
        aug_num = self.aug_list.count(True)
        aug_features, aug_label = torch.zeros(datasize * aug_num, self.n_mels, 50), np.zeros(datasize * aug_num)
        for i, path in enumerate(path_list):
            # data.shape==(channel,time)
            data, _ = torchaudio.load(os.path.join(root, path))
            # 音声データを最大の長さにパディング
            if True:
                data = F.pad(data, (0, max_length - data.size(1)))
                mel_s = mel_transform(data[0]) # [D, T]
                features[i, :, :] = mel_s
                if self.aug:
                    aug_f, aug_l = self.spec_augment(mel_s, i, self.aug_list)
                    aug_features[i*aug_num:i*aug_num+aug_num, :, :] = aug_f
                    aug_label[i*aug_num:i*aug_num+aug_num] = aug_l
            else:
                mfcc = transform(data[0])
                delta_mfcc = delta(mfcc)
                mfcc_mean = torch.mean(mfcc, axis=1)
                delta_mfcc_mean = torch.mean(delta_mfcc, axis=1)
                features[i] = torch.cat([mfcc_mean, delta_mfcc_mean])
        features = torch.cat((features, aug_features))
        self.label = np.concatenate((self.label, aug_label)).astype(np.int64)
        return features
    
    def spec_augment(self, feature, idx, aug=[True, False, False, False]):
        aug_features, aug_label = torch.zeros(aug.count(True), self.n_mels, 50), torch.tensor([self.label[idx]] * aug.count(True))
        #plot_melspectrogram(feature, title="Original Melspectrogram")
        feature = feature.unsqueeze(0)
        idx = 0
        # Time Masking
        if aug[0]:
            masking = T.TimeMasking(time_mask_param=20)
            time_masked_feature = masking(feature)
            time_masked_feature = time_masked_feature.squeeze(0)
            #plot_melspectrogram(time_masked_feature, title="Time Masked Melspectrogram")
            aug_features[idx] = time_masked_feature
            idx += 1
        # Frequency Masking
        if aug[1]:
            masking = T.FrequencyMasking(freq_mask_param=15)
            freq_masked_feature = masking(feature)
            freq_masked_feature = freq_masked_feature.squeeze(0)
            #plot_melspectrogram(freq_masked_feature, title="Freq Masked Melspectrogram")
            aug_features[idx] = freq_masked_feature
            idx += 1
        # Time Stretch
        if aug[2]:
            masking = T.TimeStretch(n_freq=self.n_mels)
            # 伸縮率は75% ~ 125%の間でランダム
            rate = np.random.choice(np.arange(75,125))/100
            time_stretched_feature = masking(feature, rate)
            # なぜか複素数が返ってくるので実数に直す
            if time_stretched_feature.dtype == torch.complex64:
                time_stretched_feature = torch.abs(time_stretched_feature)
            # MLPの場合時系列データは扱えないので系列長を固定
            if time_stretched_feature.size(2) < 50:
                padding = torch.zeros(self.n_mels, 50 - time_stretched_feature.size(2))
                time_stretched_feature = torch.cat((time_stretched_feature, padding.unsqueeze(0)), dim=2)
            elif time_stretched_feature.size(2) > 50:
                time_stretched_feature = time_stretched_feature[:, :, :50]
            time_stretched_feature = time_stretched_feature.squeeze(0)
            #plot_melspectrogram(time_stretched_feature, title="Time Stretched Melspectrogram")
            aug_features[idx] = time_stretched_feature
            idx += 1
        if aug[3]:
            if np.random.choice((True,False)):
                time_masking = T.TimeMasking(time_mask_param=20)
                masked_feature = time_masking(feature)
            else:
                rate = np.random.choice(np.arange(75,125))/100
                time_stretch = T.TimeStretch(n_freq=self.n_mels)
                masked_feature = time_stretch(feature, rate)
                # なぜか複素数が返ってくるので実数に直す
                if masked_feature.dtype == torch.complex64:
                    masked_feature = torch.abs(masked_feature)
                # MLPの場合時系列データは扱えないので系列長を固定
                if masked_feature.size(2) < 50:
                    padding = torch.zeros(self.n_mels, 50 - masked_feature.size(2))
                    masked_feature = torch.cat((masked_feature, padding.unsqueeze(0)), dim=2)
                elif masked_feature.size(2) > 50:
                    masked_feature = masked_feature[:, :, :50]
            freq_masking = T.FrequencyMasking(freq_mask_param=15)
            masked_feature = freq_masking(masked_feature)
            masked_feature = masked_feature.squeeze(0)
            #plot_melspectrogram(masked_feature, title="All Masked Melspectrogram")
            aug_features[idx] = masked_feature
        return aug_features, aug_label
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, index):
        return self.features[index], self.label[index]

#@hydra.main(version_base=None, config_path="config", config_name="train")
#def main(config: DictConfig):
def main():
    if not torch.cuda.is_available():
        print("CPU")
        device = torch.device("cpu")
    else:
        print("GPU")
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        
    # check directory existence
    #if not os.path.exists(config.out_dir):
    #    os.makedirs(config.out_dir)

    # write config to yaml file
    #with open(os.path.join(config.out_dir, "config.yaml"), "w") as f:
    #    f.write(OmegaConf.to_yaml(config))

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_truth", type=str, help='テストデータの正解ファイルCSVのパス')
    parser.add_argument("--aug", type=bool, default=True, help='Data Augmentationの有無')
    args = parser.parse_args()

    # データの読み込み
    training = pd.read_csv(os.path.join(root, "training.csv"))
    
    # Dataset の作成
    train_dataset = FSDD(training["path"].values, training['label'].values, sample_rate=8000)
    
    # Train/Validation 分割
    val_size = int(len(train_dataset)*0.2)
    train_size = len(train_dataset)-val_size
    train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            torch.Generator().manual_seed(20200616))

    if args.path_to_truth:
        # Test Dataset の作成
        test = pd.read_csv(args.path_to_truth)
        test_dataset = FSDD(test["path"].values, test['label'].values, sample_rate=8000)
    else:
        test_dataset = None
        
    # DataModule の作成
    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=32,
        num_workers=4)
    
    print(f"Number of train dataset: {len(train_dataset)}")
    print(f"Number of validation dataset: {len(val_dataset)}")
    
    # モデルの構築
    model = train(input_dim=train_dataset[0][0].shape[0], output_dim=10)
    
    # 学習の設定
    trainer = pl.Trainer(max_epochs=100, gpus=1)
    
    # モデルの学習
    trainer.fit(model=model, datamodule=datamodule)
    
    # バリデーション
    trainer.validate(model=model, datamodule=datamodule)
    
    if args.path_to_truth:
        # テスト
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
