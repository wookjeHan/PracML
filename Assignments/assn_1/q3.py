import warnings
from typing import Any, Callable, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class GoogLeNet(nn.Module):
    __constants__ = ["aux_logits", "transform_input"]

    def __init__(
        self,
        num_classes: int = 10,
        aux_logits: bool = True,
        transform_input: bool = False,
        init_weights: Optional[bool] = None,
        blocks: Optional[List[Callable[..., nn.Module]]] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, DownSample]
        if init_weights is None:
            warnings.warn(
                "The default weight initialization of GoogleNet will be changed in future releases of "
                "torchvision. If you wish to keep the old behavior (which leads to long initialization times"
                " due to scipy/scipy#11299), please set init_weights=True.",
                FutureWarning,
            )
            init_weights = True
        if len(blocks) != 3:
            raise ValueError(f"blocks length should be 3 instead of {len(blocks)}")
        conv_block = blocks[0]
        inception_block = blocks[1]
        down_sample = blocks[2]
        
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        # Conv 96 Channel 3*3 filters
        # Fashion-MNIST is grayscale 
        # Done Conv/Batch/ReLU
        self.conv1 = conv_block(1, 96, kernel_size=3, stride=1, padding=1)

        self.inception3a = inception_block(96, 32, 32)
        self.inception3b = inception_block(64, 32, 48)
        self.downsample = down_sample(80, 80)

        self.inception4a = inception_block(160, 112, 48)
        self.inception4b = inception_block(160, 96, 64)
        self.inception4c = inception_block(160, 80, 80)
        self.inception4d = inception_block(160, 48, 96)
        self.downsample2 = down_sample(144, 96)

        self.inception5a = inception_block(96+144, 176, 160)
        self.inception5b = inception_block(336, 176, 160)
        
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(336, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # N x 1 x 28 x 28
        x = self.conv1(x)
        # N * 96 * 28 * 28
        x = self.inception3a(x)
        # N * 64 * 28 * 28
        x = self.inception3b(x)
        # N * 80 * 28 * 28
        x = self.downsample(x)
        # N * 160 * 14 * 14
        x = self.inception4a(x)
        # N * 160 * 14 * 14
        x = self.inception4b(x)
        # N * 160 * 14 * 14
        x = self.inception4c(x)
        # N * 160 * 14 * 14
        x = self.inception4d(x)
        # N x 144 x 14 x 14
        x = self.downsample2(x)
        # N x 240 x 7 x 7
        x = self.inception5a(x)
        # N x 336 x 7 x 7
        x = self.inception5b(x)
        # N x 336 x 7 x 7
        x = self.avgpool(x)
        # N x 336 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 336
        x = self.dropout(x)
        # N x 336
        x = self.fc(x)
        # N x 10
        return x
    
    def forward(self, x: Tensor):
        x = self._transform_input(x)
        x = self._forward(x)
        return x

class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)
        self.branch2 = conv_block(in_channels, ch3x3, kernel_size=3, padding=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)

        outputs = [branch1, branch2]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class DownSample(nn.Module):
    def __init__(
            self,
            in_channels: int,
            ch3x3: int,
            conv_block: Optional[Callable[..., nn.Module]] = None,
        ) -> None:
            super().__init__()
            if conv_block is None:
                conv_block = BasicConv2d
            self.branch1 = conv_block(in_channels, ch3x3, kernel_size=3, stride=2, padding=1)
            self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.maxpool3(x)
        outputs = [branch1, branch2]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
# Make Dataloader   
# Q3.1
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, drop_last=False)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, drop_last=False)
# Loss function
loss_fn = torch.nn.CrossEntropyLoss()
# Make Model
num_epoch = 5*10 # We have to run 5*10 epochs
model = GoogLeNet().to("cuda")
optim = torch.optim.Adam(model.parameters(), lr=1e-9)
lr_candidates = np.logspace(-9, 1, num=10) # LR Candidates
accs = []
for epoch in range(num_epoch):
    print(f"epoch : {epoch}/50")
    for data in tqdm(train_dataloader):
        optim.zero_grad()
        batch, gts = data
        batch = batch.to("cuda")
        gts = gts.to("cuda")
        preds = model(batch)
        loss = loss_fn(preds, gts)
        loss.backward()
        optim.step()
    if(epoch%5==4):
        print("TESTING....")
        correct = 0
        total = 0
        model.eval()
        for data in tqdm(test_dataloader):
            batch, gts = data
            batch = batch.to("cuda")
            gts = gts.to("cuda")
            preds = model(batch)
            preds = torch.argmax(preds, dim=1)
            total += batch.shape[0]
            correct += (preds==gts).sum().item()
        accs.append(correct/total)
        # Change lrs
        if(epoch!=num_epoch-1):
            for g in optim.param_groups:
                g['lr'] = lr_candidates[(epoch+1)//5]
plt.plot(lr_candidates, accs, label='acc', color='red')
plt.xlabel("LR (in Log Scale)")
plt.ylabel("ACC")
plt.xscale('log')
plt.legend()
plt.title("Q3.1")
plt.savefig("Q3_1.png")
# Q3.2
lr_min = 1.0*(1e-9)
lr_max = 3.6*(1e-4)
num_epoch=15
# Let's train
model = GoogLeNet().to("cuda")
optim = torch.optim.Adam(model.parameters(), lr=lr_min)
scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=lr_min, max_lr=lr_max, step_size_up=2*len(train_dataloader) , mode="exp_range", gamma=0.99994)

train_loss = 0
validation_loss = 0 
accs = []
train_losses = []
validation_losses = []
for epoch in range(num_epoch):
    print(f"epoch : {epoch}/15")
    for data in tqdm(train_dataloader):
        model.train()
        optim.zero_grad()
        batch, gts = data
        batch = batch.to("cuda")
        gts = gts.to("cuda")
        preds = model(batch)
        loss = loss_fn(preds, gts)
        train_loss += loss.item()
        loss.backward()
        optim.step()
        # Need it for Scheduler
        scheduler.step()
        # Let's Test
    correct = 0
    total = 0
    model.eval()
    for data in tqdm(test_dataloader):
        batch, gts = data
        batch = batch.to("cuda")
        gts = gts.to("cuda")
        preds = model(batch)
        loss = loss_fn(preds, gts)
        validation_loss += loss.item()

        preds = torch.argmax(preds, dim=1)

        total += batch.shape[0]
        correct += (preds==gts).sum().item()
    accs.append(correct/total)
    train_losses.append(train_loss/len(train_dataloader))
    validation_losses.append(validation_loss/len(test_dataloader))
    train_loss = 0
    validation_loss = 0
iterations=[938 * i for i in range(15)]
plt.clf()
plt.plot(iterations, accs, label="Acc", color='red')
plt.plot(iterations, validation_losses, label="validation loss", color='blue')
plt.plot(iterations, train_losses, label="train loss", color='green')
plt.xlabel("Iteration")
plt.legend()
plt.title("Q3.2")
plt.savefig("Q3_2.png")
# Q3.3
lr = 3.6*(1e-4) # Setting to max_lr
batch_size = 32
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=False) # Starting with bn=32
model = GoogLeNet().to("cuda")
optim = torch.optim.Adam(model.parameters(), lr=lr)
train_loss = 0
train_losses = []
num_epoch = 16 # (2 epoch for 9 lr (32~4096))
for epoch in range(num_epoch):
    print(f"epoch : {epoch}/15")
    for data in tqdm(train_dataloader):
        model.train()
        optim.zero_grad()
        batch, gts = data
        batch = batch.to("cuda")
        gts = gts.to("cuda")
        preds = model(batch)
        loss = loss_fn(preds, gts)
        train_loss += loss.item()
        loss.backward()
        optim.step()
    train_losses.append(train_loss/len(train_dataloader))
    train_loss = 0
    # We need to change train_dataloader for each two epoch
    if(epoch%2==1):
        batch_size *= 2
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=False)
iterations = [1875, 1875*2, 938, 938*2, 469, 469*2, 235, 235*2, 118, 118*2, 59, 59*2, 30, 30*2, 15, 15*2]
for i in range(len(iterations)-1):
    iterations[i+1] = iterations[i]+iterations[i+1]
plt.clf()
plt.plot(iterations, train_losses, label='training_loss', color='red')
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.legend()
plt.title("Q3.3")
plt.savefig("Q3_3.png")