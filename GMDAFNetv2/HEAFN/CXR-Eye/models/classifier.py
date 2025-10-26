from typing import Optional
import torch
import torch.nn as nn
from torch.autograd import Variable
from segmentation_models_pytorch.base import ClassificationHead, SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import initialization as init
import torchvision
import torch.nn.functional as F
from .utils import N_list, stride, iou_threshs, coordinates_cat, window_nums_sum, ratios, nms
import numpy as np
import collections
import matplotlib.pyplot as plt
import cv2




def smooth_attention_map(attention_map, sigma=10.0):

    attention_map = attention_map.unsqueeze(1)  # (batch_size, 1, H, W)


    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1


    center = kernel_size // 2
    x = torch.arange(kernel_size, dtype=torch.float32, device=attention_map.device)
    x = x - center
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()


    kernel_2d = kernel_1d.view(1, 1, kernel_size, 1) * kernel_1d.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_2d.expand(1, 1, kernel_size, kernel_size)


    padding = kernel_size // 2
    padded = F.pad(attention_map, (padding, padding, padding, padding), mode='reflect')
    blurred = F.conv2d(padded, kernel_2d, padding=0, groups=1)


    blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-8)

    return blurred.squeeze(1)  # (batch_size, H, W)


def visualize_with_binary_masks(images, attention_maps, binary_masks, masked_images, page_size=8, show=True):

    if not show:
        return

    batch_size = images.shape[0]
    num_pages = (batch_size + page_size - 1) // page_size

    images_np = images.permute(0, 2, 3, 1).cpu().numpy()
    masked_images_np = masked_images.permute(0, 2, 3, 1).cpu().numpy()
    attention_maps_np = attention_maps.cpu().numpy()
    binary_masks_np = binary_masks.cpu().numpy()

    for page in range(num_pages):
        start = page * page_size
        end = min(start + page_size, batch_size)
        num_rows = end - start

        fig, axes = plt.subplots(num_rows, 4, figsize=(16, 4 * num_rows))
        if num_rows == 1:
            axes = [axes]

        for i, idx in enumerate(range(start, end)):
            ax_orig, ax_attn, ax_mask, ax_masked = axes[i]

            ax_orig.imshow(images_np[idx])
            ax_orig.set_title(f"原始图像 {idx + 1}")
            ax_orig.axis("off")

            ax_attn.imshow(attention_maps_np[idx], cmap="jet")
            ax_attn.set_title(f"边缘注意力图 {idx + 1}")
            ax_attn.axis("off")

            ax_mask.imshow(binary_masks_np[idx], cmap="gray")
            ax_mask.set_title(f"二值掩码 {idx + 1}")
            ax_mask.axis("off")

            ax_masked.imshow(masked_images_np[idx])
            ax_masked.set_title(f"掩码后图像 {idx + 1}")
            ax_masked.axis("off")

        plt.tight_layout()
        plt.show()


def extract_edge_attention(input_tensors, method='canny', threshold_low=50, threshold_high=150):

    batch_size = input_tensors.shape[0]
    device = input_tensors.device
    attention_maps = []
    

    images_np = input_tensors.cpu().numpy()
    
    for i in range(batch_size):

        img = images_np[i].transpose(1, 2, 0)  # (3, 224, 224) -> (224, 224, 3)
        

        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if method == 'canny':

            edges = cv2.Canny(gray, threshold_low, threshold_high)
            

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
        elif method == 'sobel':

            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = (edges / edges.max() * 255).astype(np.uint8)
            
        elif method == 'laplacian':

            edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
            edges = np.absolute(edges)
            edges = (edges / edges.max() * 255).astype(np.uint8)
            
        elif method == 'combined':

            canny = cv2.Canny(gray, threshold_low, threshold_high)
            
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            sobel = (sobel / sobel.max() * 255).astype(np.uint8)
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
            laplacian = np.absolute(laplacian)
            laplacian = (laplacian / laplacian.max() * 255).astype(np.uint8)
            

            edges = (0.5 * canny + 0.3 * sobel + 0.2 * laplacian).astype(np.uint8)
        
        else:
            raise ValueError(f"未知的边缘检测方法: {method}")
        

        attention = cv2.GaussianBlur(edges, (15, 15), 5)
        

        attention = attention.astype(np.float32) / 255.0
        

        attention = np.power(attention, 0.5)
        

        attention = 0.2 + 0.8 * attention
        

        attention_tensor = torch.from_numpy(attention).to(device)
        attention_maps.append(attention_tensor)
    
    attention_maps = torch.stack(attention_maps)
    return attention_maps


def af_fuction(x, k):
    return (1 - k * torch.tanh(torch.tensor(1.0))) * x + k * torch.tanh(x)


class Three_Branch(nn.Module):
    def __init__(self, n_classes, model_type='efficientnet', pretrain_path=None, 
                 edge_method='combined', edge_threshold_low=50, edge_threshold_high=150):

        super(Three_Branch, self).__init__()
        self.edge_method = edge_method
        self.edge_threshold_low = edge_threshold_low
        self.edge_threshold_high = edge_threshold_high
        
        if model_type == 'efficientnet':
            if pretrain_path is None:

                self.branch1 = get_encoder(
                    'timm-efficientnet-b0',
                    in_channels=3,
                    depth=5,
                    weights="imagenet")
                self.branch2 = get_encoder(
                    'timm-efficientnet-b0',
                    in_channels=3,
                    depth=5,
                    weights="imagenet")
                self.branch3 = get_encoder(
                    'timm-efficientnet-b0',
                    in_channels=3,
                    depth=5,
                    weights="imagenet")
                
                num_ftrs = self.branch1.out_channels[-1]  # 320
                
            else:

                self.branch1 = get_encoder('timm-efficientnet-b0', in_channels=3, depth=5)
                self.branch2 = get_encoder('timm-efficientnet-b0', in_channels=3, depth=5)
                self.branch3 = get_encoder('timm-efficientnet-b0', in_channels=3, depth=5)
                
                num_ftrs = self.branch1.out_channels[-1]
                new_dict = collections.OrderedDict()
                pretrained_dict = torch.load(pretrain_path)
                for k, v in pretrained_dict.items():
                    if 'encoder.' in k:
                        name = k[8:]
                        new_dict[name] = v
                
                redundant_keys = {"classifier.bias": None, "classifier.weight": None}
                new_dict.update(redundant_keys)
                self.branch1.load_state_dict(new_dict, strict=False)
                self.branch2.load_state_dict(new_dict, strict=False)
                self.branch3.load_state_dict(new_dict, strict=False)
        

        self.classifier = nn.Linear(num_ftrs * 3, n_classes)
        self.classifier1 = nn.Linear(num_ftrs, n_classes)
        self.classifier2 = nn.Linear(num_ftrs, n_classes)
        self.classifier3 = nn.Linear(num_ftrs, n_classes)
        
    
    def three_map(self, input, alpha, beta, gamma):
        k1 = af_fuction(input, alpha)
        k2 = af_fuction(k1, beta)
        k3 = af_fuction(k2, gamma)
        return k3
    
    def get_machine_attention_map(self, x, apply_soft_attention=True):

        with torch.no_grad():

            attention_maps = extract_edge_attention(
                x,
                method=self.edge_method,
                threshold_low=self.edge_threshold_low,
                threshold_high=self.edge_threshold_high
            )
            

            attention_maps = smooth_attention_map(attention_maps, sigma=3.0)
            

            if apply_soft_attention:

                attention_maps = 0.5 + 0.5 * attention_maps
            
            return attention_maps
    
    def forward(self, x, heatmap, show_visualization=False):

        x1 = self.branch1(x)[-1]
        x1 = F.relu(x1, inplace=True)
        
        x1 = F.adaptive_avg_pool2d(x1, (1, 1))
        x1 = torch.flatten(x1, 1)
        

        x2 = self.branch2(heatmap)[-1]
        x2 = F.relu(x2, inplace=True)
        
        x2 = F.adaptive_avg_pool2d(x2, (1, 1))
        x2 = torch.flatten(x2, 1)
        

        with torch.no_grad():

            attention_maps = extract_edge_attention(
                x,
                method=self.edge_method,
                threshold_low=self.edge_threshold_low,
                threshold_high=self.edge_threshold_high
            )
            

            if len(attention_maps.shape) == 2:
                attention_maps = attention_maps.unsqueeze(0)
            

            attention_maps = smooth_attention_map(attention_maps, sigma=3.0)
            

            attention_weight = attention_maps.unsqueeze(1)
            

            x3 = x * (0.5 + 0.5 * attention_weight)
            
            if show_visualization:
                binary_masks = (attention_maps > 0.3).float()
                visualize_with_binary_masks(x, attention_maps, 
                                        binary_masks, 
                                        x3, show=True)
        
        x3 = self.branch3(x3)[-1]
        x3 = F.relu(x3, inplace=True)
        
        x3 = F.adaptive_avg_pool2d(x3, (1, 1))
        x3 = torch.flatten(x3, 1)
        
        # 分类器部分
        p1 = self.classifier1(x1)
        p2 = self.classifier2(x2)
        p3 = self.classifier3(x3)
        alpha = torch.tanh(p1)
        beta = torch.tanh(p2)
        gamma = torch.tanh(p3)
        
        out = torch.cat((x1, x2, x3), dim=-1)
        out = F.softmax(self.classifier(out), dim=1)
        out = self.three_map(out, alpha, beta, gamma)
        
        return out, x1, x2, x3, p1, p2, p3