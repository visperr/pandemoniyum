import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class LightConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=out_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv1(x)))

class ChannelGate(nn.Module):
    def __init__(self, in_channels, num_gates=2):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // 16, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // 16, in_channels * num_gates, 1, bias=False)
        self.num_gates = num_gates

    def forward(self, x):
        x_avg = F.adaptive_avg_pool2d(x, 1)
        att = self.fc2(self.relu(self.fc1(x_avg)))
        att = att.view(att.size(0), self.num_gates, att.size(1) // self.num_gates, 1, 1)
        return F.softmax(att, dim=1)

class OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels))
        self.conv2c = nn.Sequential(LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels))
        self.conv2d = nn.Sequential(LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels))
        self.gate = ChannelGate(mid_channels, num_gates=4)
        self.conv3 = Conv1x1(mid_channels, out_channels)
        self.downsample = Conv1x1(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)

        x2 = torch.stack([x2a, x2b, x2c, x2d], dim=1)
        gate_weights = self.gate(x1)
        x2 = (x2 * gate_weights).sum(dim=1)

        x3 = self.conv3(x2)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return F.relu(x3 + residual, inplace=True)

class OSNet(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv1 = ConvLayer(3, 64, 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(OSBlock(64, 64), OSBlock(64, 64))
        self.conv2 = Conv1x1(64, 128, stride=2)
        self.layer2 = nn.Sequential(OSBlock(128, 128), OSBlock(128, 128))
        self.conv3 = Conv1x1(128, 256, stride=2)
        self.layer3 = nn.Sequential(OSBlock(256, 256), OSBlock(256, 256))
        self.fc = Conv1x1(256, feature_dim)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.layer1(x)
        x = self.conv2(x)
        x = self.layer2(x)
        x = self.conv3(x)
        x = self.layer3(x)
        x = self.fc(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.view(x.size(0), -1)

class OSNetEmbedder:
    def __init__(self, model_dir: str, device="cuda", half=False):
        self.device = device
        self.half = half
        self.model = OSNet(feature_dim=512)

        weight_path = os.path.join(model_dir, "osnet_x1_0_msmt17.pth")
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        
        if not os.path.exists(weight_path):
            print("Downloading pre-trained OSNet weights...")
            hf_url = "https://huggingface.co/spaces/rachana219/MODT2/resolve/main/trackers/strongsort/deep/checkpoint/osnet_x1_0_msmt17.pth"
            torch.hub.download_url_to_file(hf_url, weight_path)

        state_dict = torch.load(weight_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model_dict = self.model.state_dict()
        filtered_state_dict = {
            k.replace("module.", ""): v 
            for k, v in state_dict.items() 
            if k.replace("module.", "") in model_dict and v.shape == model_dict[k.replace("module.", "")].shape
        }
        self.model.load_state_dict(filtered_state_dict, strict=False)
        self.model.to(self.device)
        
        if self.half:
            self.model.half()
            
        self.model.eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        if self.half:
            self.mean, self.std = self.mean.half(), self.std.half()

    @torch.inference_mode()
    def __call__(self, crops):
        if not crops:
            return np.array([])

        resized_crops = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            c_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            c_res = cv2.resize(c_rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
            resized_crops.append(c_res)

        if not resized_crops:
            return np.array([])

        batch = torch.from_numpy(np.stack(resized_crops)).permute(0, 3, 1, 2).to(self.device)
        batch = batch.half() if self.half else batch.float()
        batch = batch / 255.0
        batch = (batch - self.mean) / self.std

        features = self.model(batch)
        features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy()
