import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import mobilenet

num_classes = 2
device = torch.device('cuda:2')
image_path = './examples/demo.jpg'
output_dir = './CAM/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
pth = './pth/mobilenet.pth'

# model
model = mobilenet()
checkpoint = torch.load(pth, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint)
model.to(device)

# image
image = Image.open(image_path).convert('RGB')
W, H = image.size
tf = transforms.Compose([
    transforms.Resize((112, 224)),
    transforms.ToTensor(),
])
x = tf(image)
x = x.to(device).unsqueeze(0)

x = model.features(x)
weight = model.classifer.weight.T
n, c, h, w = x.size()
x = x.permute(0, 2, 3, 1).contiguous().view(-1, c)
logits = torch.mm(x, weight)
logits = logits.view(h, w, num_classes).contiguous()
logits = logits.detach().cpu().numpy()

print(logits.shape)

# ng_logits, ok_logits = np.split(logits, 2, axis=2)
# ng_logits = np.squeeze(ng_logits)
# ok_logits = np.squeeze(ok_logits)
ng_logits = logits[:, :, 0]
ok_logits = logits[:, :, 1]

print(ng_logits.shape, ok_logits.shape)

cam = ng_logits

cam = cv.resize(cam, (W, H))
cam = np.maximum(cam, 0)
x_min, x_max = np.min(cam), np.max(cam)
cam = (cam - x_min) / (x_max - x_min) * 255
cam = cam.astype(dtype=np.int8)

img = cv.imread(image_path)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img_rgb)
cam_img = cam
img = plt.imshow(cam_img, cmap='jet', alpha=0.3)  # 显示activation map
plt.title('pred cls: NG')  # 打印类别信息
plt.axis('off')
# plt.imshow(img)
output_path = os.path.join(output_dir, os.path.basename(image_path))
output_path = output_path.replace('jpg', 'png')  # savefig不支持jpg，要转为png
plt.savefig(output_path)
plt.close()
