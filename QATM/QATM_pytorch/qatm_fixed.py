import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
import copy
from glob import glob
import gc
from PIL import Image
import seaborn as sns

# Set matplotlib backend
plt.ion()

def color_palette(palette, n_colors):
    """Generate color palette using seaborn-style colors"""
    colors = sns.color_palette(palette, n_colors)
    return colors

def compute_score(gray, w, h):
    """Compute geometry average on score map"""
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
    # Use cv2 filter for geometric mean computation
    kernel = np.ones((h, w), np.float32) / (h * w)
    score = cv2.filter2D(gray, -1, kernel)
    return score

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, template_dir_path, image_name, thresh_csv=None, transform=None):
        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
        
        # Fix: Get list of template paths
        self.template_path = list(template_dir_path.iterdir())
        self.template_names = [str(p) for p in self.template_path]  # Fix: Add missing attribute
        self.image_name = image_name
        
        # Load image
        self.image_raw = cv2.imread(self.image_name)
        if self.image_raw is None:
            raise ValueError(f"Could not load image: {self.image_name}")
        
        # Load threshold CSV with proper dtype handling
        self.thresh_df = None
        if thresh_csv and os.path.exists(thresh_csv):
            try:
                self.thresh_df = pd.read_csv(thresh_csv, dtype={'path': str, 'thresh': str})
            except:
                # If CSV doesn't exist or has issues, create default
                print(f"Warning: Could not load {thresh_csv}, using default thresholds")
                self.thresh_df = None
        
        # Transform image
        if self.transform:
            # Convert BGR to RGB for PIL/torch compatibility
            image_rgb = cv2.cvtColor(self.image_raw, cv2.COLOR_BGR2RGB)
            self.image = self.transform(image_rgb).unsqueeze(0)
        
    def __len__(self):
        return len(self.template_path)
    
    def __getitem__(self, idx):
        template_path = str(self.template_path[idx])
        template_raw = cv2.imread(template_path)
        
        if template_raw is None:
            raise ValueError(f"Could not load template: {template_path}")
        
        # Convert BGR to RGB for consistency
        template_rgb = cv2.cvtColor(template_raw, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            template = self.transform(template_rgb)
        else:
            template = torch.from_numpy(template_rgb).permute(2, 0, 1).float()
        
        # Get threshold for this template
        thresh = 0.7
        if self.thresh_df is not None:
            matching_rows = self.thresh_df[self.thresh_df['path'] == template_path]
            if not matching_rows.empty:
                try:
                    thresh = float(matching_rows.iloc[0]['thresh'])
                except:
                    thresh = 0.7
        
        return {
            'image': self.image, 
            'image_raw': self.image_raw, 
            'image_name': self.image_name,
            'template': template.unsqueeze(0), 
            'template_name': template_path, 
            'template_h': template.size()[-2],
            'template_w': template.size()[-1],
            'thresh': thresh
        }

class Featex():
    def __init__(self, model, use_cuda):
        self.use_cuda = use_cuda
        self.feature1 = None
        self.feature2 = None
        self.model = copy.deepcopy(model.eval())
        self.model = self.model[:17]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model[2].register_forward_hook(self.save_feature1)
        self.model[16].register_forward_hook(self.save_feature2)
        
    def save_feature1(self, module, input, output):
        self.feature1 = output.detach()
    
    def save_feature2(self, module, input, output):
        self.feature2 = output.detach()
        
    def __call__(self, input, mode='big'):
        if self.use_cuda and torch.cuda.is_available():
            input = input.cuda()
        _ = self.model(input)
        if mode == 'big':
            # resize feature1 to the same size of feature2
            self.feature1 = F.interpolate(
                self.feature1, 
                size=(self.feature2.size()[2], self.feature2.size()[3]), 
                mode='bilinear', 
                align_corners=True
            )
        else:        
            # resize feature2 to the same size of feature1
            self.feature2 = F.interpolate(
                self.feature2, 
                size=(self.feature1.size()[2], self.feature1.size()[3]), 
                mode='bilinear', 
                align_corners=True
            )
        return torch.cat((self.feature1, self.feature2), dim=1)

class MyNormLayer():
    def __call__(self, x1, x2):
        bs, _, H, W = x1.size()
        _, _, h, w = x2.size()
        eps = 1e-12
        x1 = x1.view(bs, -1, H*W)
        x2 = x2.view(bs, -1, h*w)
        concat = torch.cat((x1, x2), dim=2)
        x_mean = torch.mean(concat, dim=2, keepdim=True)
        x_std = torch.std(concat, dim=2, keepdim=True)
        x1 = (x1 - x_mean) / (x_std + eps)
        x2 = (x2 - x_mean) / (x_std + eps)
        x1 = x1.view(bs, -1, H, W)
        x2 = x2.view(bs, -1, h, w)
        return [x1, x2]

class QATM():
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, x):
        batch_size, ref_row, ref_col, qry_row, qry_col = x.size()
        x = x.view(batch_size, ref_row*ref_col, qry_row*qry_col)
        xm_ref = x - torch.max(x, dim=1, keepdim=True)[0]
        xm_qry = x - torch.max(x, dim=2, keepdim=True)[0]
        confidence = torch.sqrt(F.softmax(self.alpha*xm_ref, dim=1) * F.softmax(self.alpha * xm_qry, dim=2))
        conf_values, ind3 = torch.topk(confidence, 1)
        ind1, ind2 = torch.meshgrid(torch.arange(batch_size), torch.arange(ref_row*ref_col), indexing='ij')
        ind1 = ind1.flatten()
        ind2 = ind2.flatten()
        ind3 = ind3.flatten()
        if x.is_cuda:
            ind1 = ind1.cuda()
            ind2 = ind2.cuda()
        
        values = confidence[ind1, ind2, ind3]
        values = torch.reshape(values, [batch_size, ref_row, ref_col, 1])
        return values

class CreateModel():
    def __init__(self, alpha, model, use_cuda):
        self.alpha = alpha
        self.featex = Featex(model, use_cuda)
        self.I_feat = None
        self.I_feat_name = None
        
    def __call__(self, template, image, image_name):
        T_feat = self.featex(template)
        if self.I_feat_name != image_name:
            self.I_feat = self.featex(image)
            self.I_feat_name = image_name
        conf_maps = None
        batchsize_T = T_feat.size()[0]
        for i in range(batchsize_T):
            T_feat_i = T_feat[i].unsqueeze(0)
            I_feat_norm, T_feat_i = MyNormLayer()(self.I_feat, T_feat_i)
            dist = torch.einsum("xcab,xcde->xabde", 
                              I_feat_norm / torch.norm(I_feat_norm, dim=1, keepdim=True), 
                              T_feat_i / torch.norm(T_feat_i, dim=1, keepdim=True))
            conf_map = QATM(self.alpha)(dist)
            if conf_maps is None:
                conf_maps = conf_map
            else:
                conf_maps = torch.cat([conf_maps, conf_map], dim=0)
        return conf_maps

def nms(score, w_ini, h_ini, thresh=0.7):
    dots = np.array(np.where(score > thresh*score.max()))
    
    if dots.size == 0:
        return np.array([]).reshape(0, 2, 2)
    
    x1 = dots[1] - w_ini//2
    x2 = x1 + w_ini
    y1 = dots[0] - h_ini//2
    y2 = y1 + h_ini

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = score[dots[0], dots[1]]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= 0.5)[0]
        order = order[inds + 1]
    
    if len(keep) == 0:
        return np.array([]).reshape(0, 2, 2)
        
    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2, 0, 1)
    return boxes

def plot_result(image_raw, boxes, show=False, save_name=None, color=(255, 0, 0)):
    d_img = image_raw.copy()
    for box in boxes:
        d_img = cv2.rectangle(d_img, tuple(box[0]), tuple(box[1]), color, 3)
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    if save_name:
        cv2.imwrite(save_name, d_img)
    return d_img

def nms_multi(scores, w_array, h_array, thresh_list):
    if len(scores) == 0:
        return np.array([]).reshape(0, 2, 2), np.array([])
        
    indices = np.arange(scores.shape[0])
    maxes = np.max(scores.reshape(scores.shape[0], -1), axis=1)
    
    # omit not-matching templates
    valid_mask = maxes > 0.1 * maxes.max()
    scores_omit = scores[valid_mask]
    indices_omit = indices[valid_mask]
    
    if len(scores_omit) == 0:
        return np.array([]).reshape(0, 2, 2), np.array([])
    
    # extract candidate pixels from scores
    dots = None
    dots_indices = None
    for index, score in zip(indices_omit, scores_omit):
        dot = np.array(np.where(score > thresh_list[index]*score.max()))
        if dot.size > 0:
            if dots is None:
                dots = dot
                dots_indices = np.ones(dot.shape[-1]) * index
            else:
                dots = np.concatenate([dots, dot], axis=1)
                dots_indices = np.concatenate([dots_indices, np.ones(dot.shape[-1]) * index], axis=0)
    
    if dots is None or dots.size == 0:
        return np.array([]).reshape(0, 2, 2), np.array([])
        
    dots_indices = dots_indices.astype(np.int64)
    x1 = dots[1] - w_array[dots_indices]//2
    x2 = x1 + w_array[dots_indices]
    y1 = dots[0] - h_array[dots_indices]//2
    y2 = y1 + h_array[dots_indices]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores_vals = scores[dots_indices, dots[0], dots[1]]
    order = scores_vals.argsort()[::-1]
    dots_indices = dots_indices[order]
    
    keep = []
    keep_index = []
    while order.size > 0:
        i = order[0]
        index = dots_indices[0]
        keep.append(i)
        keep_index.append(index)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= 0.05)[0]
        order = order[inds + 1]
        dots_indices = dots_indices[inds + 1]
    
    if len(keep) == 0:
        return np.array([]).reshape(0, 2, 2), np.array([])
        
    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2,0,1)
    return boxes, np.array(keep_index)

def plot_result_multi(image_raw, boxes, indices, show=False, save_name=None, color_list=None):
    d_img = image_raw.copy()
    if len(boxes) == 0:
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB))
            plt.title("No matches found")
            plt.axis('off')
            plt.show()
        return d_img
        
    if color_list is None:
        color_list = color_palette("hls", indices.max()+1)
        color_list = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), color_list))
    
    for i in range(len(indices)):
        d_img = plot_result(d_img, boxes[i][None, :,:].copy(), color=color_list[indices[i]])
    
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    if save_name:
        cv2.imwrite(save_name, d_img)
    return d_img

def run_one_sample(model, template, image, image_name):
    val = model(template, image, image_name)
    if val.is_cuda:
        val = val.cpu()
    val = val.numpy()
    val = np.log(np.maximum(val, 1e-12))  # Prevent log(0)
    
    batch_size = val.shape[0]
    scores = []
    for i in range(batch_size):
        # compute geometry average on score map
        gray = val[i,:,:,0]
        gray = cv2.resize(gray, (image.size()[-1], image.size()[-2]))
        h = template.size()[-2]
        w = template.size()[-1]
        score = compute_score(gray, w, h)
        score[score > -1e-7] = score.min()
        score = np.exp(score / (h*w))  # reverse number range back after computing geometry average
        scores.append(score)
    return np.array(scores)

def run_multi_sample(model, dataset):
    scores = None
    w_array = []
    h_array = []
    thresh_list = []
    for i, data in enumerate(dataset):
        print(f"Processing template {i+1}/{len(dataset)}: {data['template_name']}")
        score = run_one_sample(model, data['template'], data['image'], data['image_name'])
        if scores is None:
            scores = score
        else:
            scores = np.concatenate([scores, score], axis=0)
        w_array.append(data['template_w'])
        h_array.append(data['template_h'])
        thresh_list.append(data['thresh'])
    return np.array(scores), np.array(w_array), np.array(h_array), thresh_list

# Example usage
def main():
    # Create directories if they don't exist
    os.makedirs('template', exist_ok=True)
    os.makedirs('sample', exist_ok=True)
    os.makedirs('result', exist_ok=True)
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")
    
    template_dir = 'template/'
    image_path = 'sample/sample1.jpg'
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return
    
    if not os.path.exists(template_dir) or len(list(Path(template_dir).iterdir())) == 0:
        print(f"Error: Template directory {template_dir} is empty or doesn't exist!")
        return
    
    # Create dataset
    try:
        dataset = ImageDataset(Path(template_dir), image_path, thresh_csv='thresh_template.csv')
        print(f"Loaded {len(dataset)} templates")
        
        # Create model
        model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=25, use_cuda=use_cuda)
        
        # Run detection
        print("Running template matching...")
        scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
        
        # Apply NMS
        print("Applying NMS...")
        boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)
        
        # Plot results
        print("Plotting results...")
        d_img = plot_result_multi(dataset.image_raw, boxes, indices, show=True, save_name='result/result_sample.png')
        
        print(f"Found {len(boxes)} matches")
        print("Results saved to result/result_sample.png")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()