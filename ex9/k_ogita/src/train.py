#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
B4輪講最終課題 パターン認識に挑戦してみよう
特徴量；メルスペクトログラム
識別器；CNN
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

import os
from logging import getLogger

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchmetrics
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, random_split

logger = getLogger(__name__)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../"

class CNN(nn.Module):
    """CNN model."""

    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 10 * 6, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.2)

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


class MLP(nn.Module):
    """MLP model."""

    def __init__(self, input_dim, output_dim, feature, frame_lengths):
        super(MLP, self).__init__()
        self.flatten = torch.nn.Flatten()
        if feature == "melspc":
            self.fc1 = torch.nn.Linear(input_dim * frame_lengths, 256)
        else:
            self.fc1 = torch.nn.Linear(input_dim, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class train(pl.LightningModule):
    """Train model."""

    def __init__(
        self, input_dim, output_dim, model, feature, lr=0.002, frame_lengths=None
    ):
        super().__init__()
        if model == "MLP":
            self.model = MLP(input_dim, output_dim, feature, frame_lengths)
        else:
            self.model = CNN(input_dim, output_dim)
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.confm = torchmetrics.ConfusionMatrix(10, normalize="true")
        print(self.model)

    def forward(self, x):
        return self.model(x.unsqueeze(1))

    def training_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/acc",
            self.train_acc(pred, y),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log("val/acc", self.val_acc(pred, y), prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log("test/acc", self.test_acc(pred, y), prog_bar=True, logger=True)
        return {"pred": torch.argmax(pred, dim=-1), "target": y}

    def test_epoch_end(self, outputs) -> None:
        # 混同行列を tensorboard に出力
        preds = torch.cat([tmp["pred"] for tmp in outputs])
        targets = torch.cat([tmp["target"] for tmp in outputs])
        confusion_matrix = self.confm(preds, targets)
        df_cm = pd.DataFrame(
            confusion_matrix.cpu().numpy(), index=range(10), columns=range(10)
        )
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="gray_r").get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return self.optimizer


class FSDD(Dataset):
    """
    Create dataset.

    Args:
        Dataset : dataset.
    """

    def __init__(
        self,
        path_list,
        label,
        sample_rate,
        feature,
        fft_size,
        hop_size,
        win_length,
        window,
        pad_mode,
        frame_lengths,
        n_mels,
        fmin,
        fmax,
        n_mfcc,
        aug,
        aug_list,
        time_mask_param,
        freq_mask_param,
        power,
    ) -> None:
        super(FSDD, self).__init__()
        self.label = label
        self.sample_rate = sample_rate
        self.feature = feature
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        if window == "hann":
            self.window = torch.hann_window
        else:
            self.window = torch.hamming_window
        self.pad_mode = pad_mode
        self.frame_lengths = frame_lengths
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.n_mfcc = n_mfcc
        self.power = power
        self.aug = aug
        self.aug_list = aug_list
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.features = self.feature_extraction(path_list)

    def feature_extraction(self, path_list):
        """
        Extract features from audio files.

        Args:
            path_list(list): Path list of files from which to extract features.

        Returns:
            tensor: extracted features.
        """
        datasize = len(path_list)
        if self.feature == "melspc":
            max_length = (self.frame_lengths - 1) * self.hop_size
            features = torch.zeros(datasize, self.n_mels, self.frame_lengths)
            mel_transform = T.MelSpectrogram(
                n_fft=self.fft_size,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                hop_length=self.hop_size,
                win_length=self.win_length,
                window_fn=self.window,
                pad_mode=self.pad_mode,
                power=self.power,
            )
        elif self.feature == "mfcc":
            features = torch.zeros(datasize, self.n_mfcc * 2)
            transform = torchaudio.transforms.MFCC(
                n_mfcc=self.n_mfcc,
                melkwargs={"n_mels": self.n_mels, "n_fft": self.fft_size},
            )
        else:
            raise ValueError('"The value of "feature" can only be "melspc" or "mfcc"')

        # Data Augmentation用のリスト
        aug_num = self.aug_list.count(True)
        aug_features, aug_label = torch.zeros(
            datasize * aug_num, self.n_mels, self.frame_lengths
        ), np.zeros(datasize * aug_num)
        for i, path in enumerate(path_list):
            # data.shape==(channel,time)
            data, _ = torchaudio.load(os.path.join(root, path))
            if self.feature == "melspc":
                # Padding data
                data = F.pad(data, (0, max_length - data.size(1)))
                mel_s = mel_transform(data[0])  # [D, T]
                features[i, :, :] = mel_s
                # Apply spec augmentation
                if self.aug:
                    aug_f, aug_l = self.spec_augment(mel_s, i)
                    aug_features[i * aug_num : i * aug_num + aug_num, :, :] = aug_f
                    aug_label[i * aug_num : i * aug_num + aug_num] = aug_l
            elif self.feature == "mfcc":
                mfcc = transform(data[0])
                delta_mfcc = self.delta(mfcc)
                mfcc_mean = torch.mean(mfcc, axis=1)
                delta_mfcc_mean = torch.mean(delta_mfcc, axis=1)
                features[i] = torch.cat([mfcc_mean, delta_mfcc_mean])
        if self.feature == "melspc" and self.aug:
            features = torch.cat((features, aug_features))
            self.label = np.concatenate((self.label, aug_label)).astype(np.int64)
        return features

    def delta(self, mfcc, l=2):
        """
        Compute ΔMFCC.

        Args:
            mfcc (ndarray): MFCC.
            l (int, optional): Number of frames between which the difference is taken. Defaults to 2.

        Returns:
            ndarray: ΔMFCC.
        """
        mfcc_pad = np.pad(mfcc, [[l, l + 1], [0, 0]], "edge")
        k_square = np.sum(np.power(np.arange(-l, l + 1), 2))
        k_sequence = np.arange(-l, l + 1)
        delta_mfcc = np.zeros_like(mfcc)
        for i in range(mfcc.shape[0]):
            delta_mfcc[i] = np.dot(k_sequence, mfcc_pad[i : i + l * 2 + 1])
        delta_mfcc = delta_mfcc / k_square
        return torch.from_numpy(delta_mfcc.astype(np.float32)).clone()

    def spec_augment(self, feature, idx):
        """
        Apply spec augmentation.

        Args:
            feature (tensor): features.
            idx (int): index of feature.

        Returns:
            tensor: Data obtained from data augmentation.
        """
        # Make lists for data augmentation
        aug_features, aug_label = torch.zeros(
            self.aug_list.count(True), self.n_mels, self.frame_lengths
        ), torch.tensor([self.label[idx]] * self.aug_list.count(True))
        feature = feature.unsqueeze(0)
        idx = 0
        # Time masking
        if self.aug_list[0]:
            masking = T.TimeMasking(time_mask_param=self.time_mask_param)
            time_masked_feature = masking(feature)
            time_masked_feature = time_masked_feature.squeeze(0)
            aug_features[idx] = time_masked_feature
            idx += 1
        # Frequency masking
        if self.aug_list[1]:
            masking = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)
            freq_masked_feature = masking(feature)
            freq_masked_feature = freq_masked_feature.squeeze(0)
            aug_features[idx] = freq_masked_feature
            idx += 1
        # Time stretch
        if self.aug_list[2]:
            masking = T.TimeStretch(n_freq=self.n_mels)
            # 伸縮率は75% ~ 125%の間でランダム
            rate = np.random.choice(np.arange(75, 125)) / 100
            time_stretched_feature = masking(feature, rate)
            # なぜか複素数が返ってくるので実数に直す
            if time_stretched_feature.dtype == torch.complex64:
                time_stretched_feature = torch.abs(time_stretched_feature)
            if time_stretched_feature.size(2) < self.frame_lengths:
                padding = torch.zeros(
                    self.n_mels, self.frame_lengths - time_stretched_feature.size(2)
                )
                time_stretched_feature = torch.cat(
                    (time_stretched_feature, padding.unsqueeze(0)), dim=2
                )
            elif time_stretched_feature.size(2) > self.frame_lengths:
                time_stretched_feature = time_stretched_feature[:, :, :self.frame_lengths]
            time_stretched_feature = time_stretched_feature.squeeze(0)
            aug_features[idx] = time_stretched_feature
            idx += 1
        # (Time masking or time stretch) & freq masking
        if self.aug_list[3]:
            if np.random.choice((True, False)):
                time_masking = T.TimeMasking(time_mask_param=self.time_mask_param)
                masked_feature = time_masking(feature)
            else:
                rate = np.random.choice(np.arange(75, 125)) / 100
                time_stretch = T.TimeStretch(n_freq=self.n_mels)
                masked_feature = time_stretch(feature, rate)
                # なぜか複素数が返ってくるので実数に直す
                if masked_feature.dtype == torch.complex64:
                    masked_feature = torch.abs(masked_feature)
                if masked_feature.size(2) < self.frame_lengths:
                    padding = torch.zeros(
                        self.n_mels, self.frame_lengths - masked_feature.size(2)
                    )
                    masked_feature = torch.cat(
                        (masked_feature, padding.unsqueeze(0)), dim=2
                    )
                elif masked_feature.size(2) > self.frame_lengths:
                    masked_feature = masked_feature[:, :, : self.frame_lengths]
            freq_masking = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)
            masked_feature = freq_masking(masked_feature)
            masked_feature = masked_feature.squeeze(0)
            aug_features[idx] = masked_feature
        return aug_features, aug_label

    def __len__(self):
        """
        Return the number of features.

        Returns:
            int: number of features.
        """
        return self.features.shape[0]

    def __getitem__(self, index):
        """
        Return features and labels.

        Args:
            index (int): index.

        Returns:
            tuple: features and labels.
        """
        return self.features[index], self.label[index]


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config: DictConfig):
    if not torch.cuda.is_available():
        print("CPU")
    else:
        print("GPU")
        torch.backends.cudnn.benchmark = True

    # check directory existence
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    # write config to yaml file
    with open(os.path.join(config.out_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config))

    # データの読み込み
    training = pd.read_csv(os.path.join(root, config.path_to_training))

    dataset_params = {
        "sample_rate": config.train.sample_rate,
        "feature": config.train.feature,
        "fft_size": config.train.fft_size,
        "hop_size": config.train.hop_size,
        "win_length": config.train.win_length,
        "window": config.train.window,
        "pad_mode": config.train.pad_mode,
        "frame_lengths": config.train.frame_lengths,
        "n_mels": config.train.n_mels,
        "fmin": config.train.fmin,
        "fmax": config.train.fmax,
        "n_mfcc": config.train.n_mfcc,
        "aug": config.train.aug,
        "aug_list": config.train.aug_list,
        "time_mask_param": config.train.time_mask_param,
        "freq_mask_param": config.train.freq_mask_param,
        "power": config.train.power,
    }

    # Dataset の作成
    train_dataset = FSDD(
        training["path"].values, training["label"].values, **dataset_params
    )

    # Train/Validation 分割
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size], torch.Generator().manual_seed(20200616)
    )

    if config.path_to_truth:
        # Test Dataset の作成
        test = pd.read_csv(os.path.join(root, config.path_to_truth))
        test_dataset = FSDD(test["path"].values, test["label"].values, **dataset_params)
    else:
        test_dataset = None

    # DataModule の作成
    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
    )

    print(f"Number of train dataset: {len(train_dataset)}")
    print(f"Number of validation dataset: {len(val_dataset)}")

    if config.train.model == "CNN":
        input_dim = 1
    elif config.train.model == "MLP":
        input_dim = train_dataset[0][0].shape[0]
    else:
        raise ValueError('"The value of "model" can only be "CNN" or "MLP"')
    # モデルの構築
    model = train(
        input_dim=input_dim,
        output_dim=10,
        feature=config.train.feature,
        model=config.train.model,
        lr=config.train.lr,
        frame_lengths=config.train.frame_lengths,
    )

    # 学習の設定
    # trainer = pl.Trainer(max_epochs=config.train.max_epochs)
    trainer = pl.Trainer(max_epochs=config.train.max_epochs, gpus=1)

    # モデルの学習
    trainer.fit(model=model, datamodule=datamodule)

    # バリデーション
    trainer.validate(model=model, datamodule=datamodule)

    if config.path_to_truth:
        # テスト
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
