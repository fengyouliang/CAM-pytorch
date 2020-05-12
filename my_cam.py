import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import mobilenet

from tqdm import tqdm

# num_classes = 2
# device = torch.device('cuda:2')
# image_path = './examples/demo.jpg'
# output_dir = './CAM/'
#
# pth = './pth/mobilenet.pth'


def cam(class_name, image_path, pth, device=torch.device('cuda:2'), num_classes=2, output_dir='./CAM/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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

    class_index = 0 if class_name == 'ng' else 1
    cam = logits[:, :, class_index]
    cam = cv.resize(cam, (W, H))
    cam = np.maximum(cam, 0)
    x_min, x_max = np.min(cam), np.max(cam)
    cam = (cam - x_min) / (x_max - x_min) * 255
    cam = cam.astype(dtype=np.int8)

    img = cv.imread(image_path)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(img_rgb)
    plt.axis('off')
    cam_img = cam
    plt.imshow(cam_img, cmap='jet', alpha=0.3)  # 显示activation map
    plt.subplot(2, 1, 2)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f'class: {class_name}')  # 打印类别信息
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    output_path = output_path.replace('jpg', 'png')  # savefig不支持jpg，要转为png
    plt.savefig(output_path)
    plt.close()

    # plt.axis('off')
    # plt.gcf().set_size_inches(W / 100, H / 100)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # output_path = os.path.join(output_dir, os.path.basename(image_path))
    # output_path = output_path.replace('jpg', 'png')  # savefig不支持jpg，要转为png
    # plt.savefig(output_path)


def cam_test():
    pth = '/home/youliang/code/binary/best_FNR_model/31_acc_0.9443_mAP_0.9620500000000001_FNR0.5503.pth'
    root_path = '/mnt/tmp/feng/kuozhankuang/fold_1/test'
    mode = 'ng'
    test_path = f'{root_path}/{mode}'
    bar = tqdm(os.listdir(test_path))
    for file in bar:
        bar.set_description(file)
        image_path = f'{test_path}/{file}'
        cam('ng', image_path, pth, output_dir=f'./CAM/test/{mode}')


if __name__ == '__main__':
    # cam('ng', './examples/demo.jpg', './pth/mobilenet.pth')
    cam_test()
