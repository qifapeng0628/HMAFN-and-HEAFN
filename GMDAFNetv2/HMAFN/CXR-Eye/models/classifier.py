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
#from utils.dataset import heatmap_show, imshow
import collections
from torchvision import models
from timm import create_model
from contextlib import contextmanager
import matplotlib.pyplot as plt

try:
    import torchxrayvision as xrv
except ImportError:
    print("请安装torchxrayvision: pip install torchxrayvision")
    
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_cls_attention_map(model, x, first_n_layers=3, use_first_layers=True):

    attn_maps = []

    def hook_fn(module, input, output):
        attn_maps.append(output)  # (batch_size, num_heads, num_tokens, num_tokens)

    hooks = [block.attn.attn_drop.register_forward_hook(hook_fn) for block in model.blocks]

    with torch.no_grad():
        _ = model(x)

    for hook in hooks:
        hook.remove()


    if use_first_layers:
        selected_attn_maps = attn_maps[:first_n_layers]
    else:
        selected_attn_maps = attn_maps[-first_n_layers:]

    cls_attn_list = []
    for attn_map in selected_attn_maps:
        attn_map = attn_map.mean(dim=1)
        cls_attn = attn_map[:, 0, 1:]
        cls_attn_list.append(cls_attn)

    avg_cls_attn = torch.stack(cls_attn_list).mean(dim=0)
    avg_cls_attn = avg_cls_attn.reshape(x.shape[0], 14, 14)

    return avg_cls_attn



def process_attention_maps(attention_maps):
    processed_maps = torch.nn.functional.interpolate(attention_maps.unsqueeze(1), size=(224, 224), mode='bilinear',
                                                     align_corners=False).squeeze(1)
    processed_maps = (processed_maps - processed_maps.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / (
                processed_maps.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] -
                processed_maps.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0])
    return processed_maps


def visualize_attention(images, attention_maps, R2_images, page_size=8, show=True):
    if not show:
        return

    batch_size = images.shape[0]
    num_pages = (batch_size + page_size - 1) // page_size

    images_np = images.permute(0, 2, 3, 1).cpu().numpy()
    R2_images_np = R2_images.permute(0, 2, 3, 1).cpu().numpy()
    attention_maps_np = attention_maps.cpu().numpy()

    for page in range(num_pages):
        start = page * page_size
        end = min(start + page_size, batch_size)
        num_rows = end - start

        fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
        if num_rows == 1:
            axes = [axes]

        for i, idx in enumerate(range(start, end)):
            ax_orig, ax_attn, ax_r2 = axes[i]

            # 原图
            ax_orig.imshow(images_np[idx])
            ax_orig.set_title(f"Original Image {idx + 1}")
            ax_orig.axis("off")

            # 注意力图
            ax_attn.imshow(attention_maps_np[idx], cmap="jet")
            ax_attn.set_title(f"ViT CLS Attention Map {idx + 1}")
            ax_attn.axis("off")

            # 乘积后的新图像
            ax_r2.imshow(R2_images_np[idx])
            ax_r2.set_title(f"Image × (1 + Attention) {idx + 1}")
            ax_r2.axis("off")

        plt.tight_layout()
        plt.show()


def smooth_attention_map(attention_map, sigma=10.0):

    import torch
    import torch.nn.functional as F
    

    original_shape = attention_map.shape
    needs_squeeze = False
    

    if len(attention_map.shape) == 3:

        attention_map = attention_map.unsqueeze(1)
        needs_squeeze = True
    elif len(attention_map.shape) == 2:

        attention_map = attention_map.unsqueeze(0).unsqueeze(0)
        needs_squeeze = True


    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    center = kernel_size // 2
    x = torch.arange(kernel_size, dtype=torch.float32, device=attention_map.device)
    x = x - center
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.view(1, 1, kernel_size, 1) * kernel_1d.view(1, 1, 1, kernel_size)

    padding = kernel_size // 2
    padded = F.pad(attention_map, (padding, padding, padding, padding), mode='replicate')
    blurred = F.conv2d(padded, kernel_2d, padding=0, groups=1)


    blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-8)


    if needs_squeeze:
        blurred = blurred.squeeze(1)
    

    if len(original_shape) == 3:
        return blurred
    elif len(original_shape) == 2:
        return blurred.squeeze(0)
    
    return blurred

def convert_to_binary_mask(processed_attention_maps, threshold=0.5, expansion_size=100):
    binary_masks = (processed_attention_maps > threshold).float()


    kernel_size = expansion_size
    padding = kernel_size // 2
    expanded_masks = F.max_pool2d(binary_masks.unsqueeze(1), kernel_size=kernel_size, stride=1,
                                  padding=padding).squeeze(1)


    expanded_masks = F.interpolate(expanded_masks.unsqueeze(1), size=(224, 224), mode='nearest').squeeze(1)

    return expanded_masks


def apply_binary_mask_on_images(input_tensors, binary_masks):

    binary_masks = binary_masks.to(input_tensors.device)
    masked_images = input_tensors * binary_masks.unsqueeze(1)  # (batch_size, 3, 224, 224)
    return masked_images


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

            # 原图
            ax_orig.imshow(images_np[idx])
            ax_orig.set_title(f"原始图像 {idx + 1}")
            ax_orig.axis("off")

            # 注意力图
            ax_attn.imshow(attention_maps_np[idx], cmap="jet")
            ax_attn.set_title(f"注意力图 {idx + 1}")
            ax_attn.axis("off")

            # 二值掩码
            ax_mask.imshow(binary_masks_np[idx], cmap="gray")
            ax_mask.set_title(f"二值掩码 {idx + 1}")
            ax_mask.axis("off")

            # 掩码后的图像
            ax_masked.imshow(masked_images_np[idx])
            ax_masked.set_title(f"掩码后图像 {idx + 1}")
            ax_masked.axis("off")

        plt.tight_layout()
        plt.show()


# ============ 改进的注意力过滤函数 ============
def improved_attention_filter(model, input_tensors, last_n_layers=3, use_first_layers=True, method='gaussian',
                              threshold=0.3, show_visualization=False):

    import torch.nn.functional as F

    attention_maps = get_cls_attention_map(model, input_tensors, last_n_layers, use_first_layers=use_first_layers)
    attention_maps = process_attention_maps(attention_maps)


    if method == 'gaussian':
        processed_maps = smooth_attention_map(attention_maps, sigma=5.0)
        binary_masks = (processed_maps > threshold).float()
    elif method == 'original':

        binary_masks = convert_to_binary_mask(attention_maps, threshold=threshold, expansion_size=100)
    else:
        raise ValueError(f"未知的方法: {method}")


    masked_images = apply_binary_mask_on_images(input_tensors, binary_masks)


    if show_visualization:
        visualize_with_binary_masks(input_tensors, attention_maps, binary_masks, masked_images, show=True)

    return masked_images


def af_binary(model, input_tensors, last_n_layers=3, use_first_layers=True, threshold=0.1, show_visualization=False):


    attention_maps = get_cls_attention_map(model, input_tensors, last_n_layers, use_first_layers=use_first_layers)


    processed_attention_maps = process_attention_maps(attention_maps)


    binary_masks = convert_to_binary_mask(processed_attention_maps, threshold=threshold)


    masked_images = apply_binary_mask_on_images(input_tensors, binary_masks)


    if show_visualization:
        visualize_with_binary_masks(input_tensors, processed_attention_maps, binary_masks, masked_images, show=True)

    return masked_images




def extract_medical_attention_simple(model, input_tensors):

    batch_size = input_tensors.shape[0]
    device = input_tensors.device
    

    model.eval()
    

    attention_maps = []
    

    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    

    hooks = []
    target_layers = ['features.denseblock3', 'features.denseblock4', 'features.transition2']
    for target in target_layers:
        for name, module in model.named_modules():
            if target in name and 'conv' in name:
                h = module.register_forward_hook(get_activation(name))
                hooks.append(h)
                break
    
    with torch.no_grad():
        for i in range(batch_size):

            single_input = input_tensors[i:i+1]
            


            input_min = single_input.min()
            input_max = single_input.max()
            

            if input_max > 1.5:

                single_input = single_input / 255.0

                single_input = (single_input * 2 - 1) * 1024
            elif input_min >= 0 and input_max <= 1.0:

                single_input = (single_input * 2 - 1) * 1024
            elif input_min >= -1.0 and input_max <= 1.0:

                single_input = single_input * 1024
            else:


                single_input = (single_input - input_min) / (input_max - input_min + 1e-8)

                single_input = (single_input * 2 - 1) * 1024
            

            single_input = torch.clamp(single_input, -1024, 1024)
            

            activations.clear()

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    _ = model(single_input)
                except Exception as e:
                    print(f"模型前向传播错误: {e}")
                    single_input = input_tensors[i:i+1]
                    _ = model(single_input)
            

            if len(activations) > 0:
                attention = None
                for idx, (name, activation) in enumerate(activations.items()):

                    act_mean = activation.mean(dim=1, keepdim=True)
                    act_var = activation.var(dim=1, keepdim=True)
                    

                    current_attention = torch.sqrt(act_mean.pow(2) + act_var)

                    if attention is not None:

                        target_h = max(attention.shape[2], current_attention.shape[2])
                        target_w = max(attention.shape[3], current_attention.shape[3])
                        

                        if attention.shape[2] != target_h or attention.shape[3] != target_w:
                            attention = F.interpolate(attention, size=(target_h, target_w), 
                                                    mode='bilinear', align_corners=False)
                        if current_attention.shape[2] != target_h or current_attention.shape[3] != target_w:
                            current_attention = F.interpolate(current_attention, size=(target_h, target_w), 
                                                            mode='bilinear', align_corners=False)
                    

                    current_attention = current_attention - current_attention.min()
                    current_attention = current_attention / (current_attention.max() + 1e-8)
                    
                    if attention is None:
                        attention = current_attention
                    else:

                        weight = (idx + 1) / len(activations)
                        attention = attention * (1 - weight * 0.3) + current_attention * weight * 0.3
            else:

                x = single_input
                for name, layer in model.features._modules.items():
                    x = layer(x)
                

                attention_l2 = torch.norm(x, p=2, dim=1, keepdim=True)
                attention_var = x.var(dim=1, keepdim=True)
                attention = attention_l2 * 0.5 + attention_var * 0.5
                

                attention = attention - attention.min()
                attention = attention / (attention.max() + 1e-8)
            


            if attention.shape[2] >= 3 and attention.shape[3] >= 3:
                attention = F.avg_pool2d(attention, kernel_size=3, stride=1, padding=1)
            

            attention = torch.sigmoid((attention - 0.5) * 8)
            

            attention = F.interpolate(
                attention, 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            )
            
            attention_maps.append(attention.squeeze(1))
    

    for hook in hooks:
        hook.remove()

    attention_maps = torch.stack(attention_maps)
    
    return attention_maps

def extract_medical_attention_gradcam(model, input_tensors, target_layer_name='features.transition3.conv'):

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    
    batch_size = input_tensors.shape[0]
    device = input_tensors.device
    attention_masks = []
    

    processed_input = input_tensors.clone()
    if processed_input.max() > 1:
        processed_input = processed_input / 255.0
    processed_input = processed_input * 2048 - 1024
    

    target_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break
    
    # 备选层
    if target_layer is None:
        possible_layers = [
            'features.transition3.conv',
            'features.transition2.conv', 
            'features.denseblock4.denselayer16.conv2',
            'features.denseblock3.denselayer24.conv2'
        ]
        for layer_name in possible_layers:
            for name, module in model.named_modules():
                if name == layer_name:
                    target_layer = module
                    break
            if target_layer is not None:
                break
    
    if target_layer is None:

        return extract_medical_attention_simple(model, input_tensors)
    

    try:
        cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device.type == 'cuda'))
    except:
        cam = GradCAM(model=model, target_layers=[target_layer])
    

    for i in range(batch_size):
        img_tensor = processed_input[i:i+1]
        
        try:

            with torch.no_grad():
                output = model(img_tensor)
                pred_class = output.argmax(dim=1).item()
            

            targets = [ClassifierOutputTarget(pred_class)]
            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
            

            attention = torch.from_numpy(grayscale_cam[0]).to(device)
            

            attention = torch.sigmoid((attention - 0.5) * 5)
            
            attention_masks.append(attention)
            
        except Exception as e:
            print(f"GradCAM计算错误: {e}")

            fallback = extract_medical_attention_simple(model, img_tensor)
            attention_masks.append(fallback.squeeze(0))
    
    attention_masks = torch.stack(attention_masks)
    

    del cam
    
    return attention_masks


def apply_medical_attention_mask(input_tensors, attention_maps, threshold=0.3, smooth_sigma=5.0):


    if smooth_sigma > 0:
        attention_maps = smooth_attention_map(attention_maps, sigma=smooth_sigma)
    

    binary_masks = (attention_maps > threshold).float()
    

    masked_images = input_tensors * binary_masks.unsqueeze(1)
    
    return masked_images

def af_fuction(x,k):
    return (1-k*torch.tanh(torch.tensor(1.0)))*x + k*torch.tanh(x)
    


class Three_Branch(nn.Module):
    def __init__(self, n_classes, model_type='efficientnet', pretrain_path=None, 
                 use_medical_attention=False, medical_model_weights='densenet121-res224-mimic_ch',
                 use_gradcam=False):
        super(Three_Branch, self).__init__()
        self.use_medical_attention = use_medical_attention
        self.use_gradcam = use_gradcam
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
                

                if use_medical_attention:
                    try:

                        self.medical_attention_model = xrv.models.DenseNet(weights=medical_model_weights)

                        for name, param in self.medical_attention_model.named_parameters():
                            if 'features.conv0' in name or 'features.norm0' in name or 'features.denseblock1' in name:
                                param.requires_grad = False

                        self.medical_attention_model.eval()
                        print(f"成功加载医疗预训练模型: {medical_model_weights}")
                    except Exception as e:
                        print(f"加载医疗模型失败，使用原始ViT: {e}")
                        self.use_medical_attention = False

                        self.model = create_model('vit_base_patch16_224_in21k', 
                                                pretrained=False,
                                                checkpoint_path='/root/.cache/torch/hub/checkpoints/vit_base_patch16_224_in21k/pytorch_model.bin')
                else:

                    self.model = create_model('vit_base_patch16_224_in21k', 
                                            pretrained=False,
                                            checkpoint_path='/root/.cache/torch/hub/checkpoints/vit_base_patch16_224_in21k/pytorch_model.bin')
                
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
                

                if use_medical_attention:
                    try:
                        self.medical_attention_model = xrv.models.DenseNet(weights=medical_model_weights)
                        for name, param in self.medical_attention_model.named_parameters():
                            if 'features.conv0' in name or 'features.norm0' in name or 'features.denseblock1' in name:
                                param.requires_grad = False
                        self.medical_attention_model.eval()
                    except:
                        self.use_medical_attention = False
        

        self.classifier = nn.Linear(num_ftrs * 3, n_classes)
        self.classifier1 = nn.Linear(num_ftrs,n_classes)
        self.classifier2 = nn.Linear(num_ftrs,n_classes)
        self.classifier3 = nn.Linear(num_ftrs,n_classes)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
    
    def three_map(self,input,alpha,beta,gamma):
        k1 = af_fuction(input,alpha)
        k2 = af_fuction(k1,beta)
        k3 = af_fuction(k2,gamma)
        return k3
    
    def get_machine_attention_map(self, x, apply_soft_attention=True):

        with torch.no_grad():
            if self.use_medical_attention:
                x_gray = x.mean(dim=1, keepdim=True)
                
                if self.use_gradcam:
                    try:
                        attention_maps = extract_medical_attention_gradcam(
                            self.medical_attention_model,
                            x_gray,
                            target_layer_name='features.transition3.conv'
                        )
                    except:
                        attention_maps = extract_medical_attention_simple(
                            self.medical_attention_model, x_gray
                        )
                else:
                    attention_maps = extract_medical_attention_simple(
                        self.medical_attention_model, x_gray
                    )
                

                if len(attention_maps.shape) == 2:
                    attention_maps = attention_maps.unsqueeze(0)
                

                attention_maps = smooth_attention_map(attention_maps, sigma=3.0)
                

                if apply_soft_attention:

                    attention_maps = 0.3 + 0.7 * attention_maps
            else:

                attention_maps = get_cls_attention_map(
                    self.model, x,
                    first_n_layers=12,
                    use_first_layers=False
                )
                attention_maps = process_attention_maps(attention_maps)
                

                attention_maps = smooth_attention_map(attention_maps, sigma=5.0)
                

                if apply_soft_attention:

                    attention_maps = 0.6 + 0.4 * attention_maps
            
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
        

        if self.use_medical_attention:

            x_gray = x.mean(dim=1, keepdim=True)
            

            if self.use_gradcam:
                try:
                    attention_maps = extract_medical_attention_gradcam(
                        self.medical_attention_model,
                        x_gray,
                        target_layer_name='features.transition3.conv'
                    )
                except Exception as e:
                    print(f"GradCAM失败: {e}")
                    attention_maps = extract_medical_attention_simple(
                        self.medical_attention_model, x_gray
                    )
            else:
                attention_maps = extract_medical_attention_simple(
                    self.medical_attention_model, x_gray
                )
            

            if len(attention_maps.shape) == 2:
                attention_maps = attention_maps.unsqueeze(0)
            

            attention_std = attention_maps.std()
            sigma = max(1.0, min(5.0, 3.0 * (1.0 - attention_std)))
            attention_maps = smooth_attention_map(attention_maps, sigma=sigma)
            

            attention_weight = attention_maps.unsqueeze(1)
            

            attention_mean = attention_weight.mean()
            alpha = 0.2 + 0.3 * torch.sigmoid(attention_mean - 0.5)
            x3 = x * (alpha + (1 - alpha) * attention_weight)
            
            if show_visualization:
                binary_masks = (attention_maps > 0.3).float()
                visualize_with_binary_masks(x, attention_maps, 
                                           binary_masks, 
                                           x3, show=True)
        else:

            with torch.no_grad():

                attention_maps = get_cls_attention_map(
                    self.model, x,
                    first_n_layers=12,
                    use_first_layers=False
                )
                attention_maps = process_attention_maps(attention_maps)
                

                attention_maps = smooth_attention_map(attention_maps, sigma=5.0)
                

                if len(attention_maps.shape) == 3:

                    batch_size = attention_maps.shape[0]
                elif len(attention_maps.shape) == 2:

                    attention_maps = attention_maps.unsqueeze(0)
                    batch_size = 1
                

                if attention_maps.shape[0] != x.shape[0]:

                    if attention_maps.shape[0] == 1 and x.shape[0] > 1:
                        attention_maps = attention_maps.expand(x.shape[0], -1, -1)
                

                attention_weight = attention_maps.unsqueeze(1)
                attention_weight = attention_weight.repeat(1, 3, 1, 1)
                

                x3 = x * (0.6 + 0.4 * attention_weight)
                
                if show_visualization:
                    binary_masks = (attention_maps > 0.3).float()
                    visualize_with_binary_masks(x, attention_maps, 
                                            binary_masks, 
                                            x3, show=True)
        x3 = self.branch3(x3)[-1]
        x3 = F.relu(x3, inplace=True)
        x3 = F.adaptive_avg_pool2d(x3, (1, 1))
        x3 = torch.flatten(x3, 1)
        

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