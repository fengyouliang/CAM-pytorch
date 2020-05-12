import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from keras.models import Model
from keras.applications.resnet50 import ResNet50
import keras

from classification_models.keras import Classifiers
from configs import FLAGS
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
      base_model: keras model excluding top
      nb_classes: # of classes
    Returns:
      new keras model with last layer
    """
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(nb_classes)(x)
    if FLAGS.datatype in ["binary", "multilabel"]:
        activation = 'sigmoid'
    elif FLAGS.datatype == "multiclass":
        activation = 'softmax'
    predictions = keras.layers.Activation(activation, name='output')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# 参考代码链接：https://github.com/alexisbcook/ResNetCAM-keras
class CAM(object):
    def __init__(self, class_dict, model_path, img_size=(224, 224, 3)):
        """构造函数，获取可视化需要的信息。
        Args：
            class_dict: {class_id1:class_name1, class_id2:class_name2, ...}
            model_path: h5模型路径，默认为只有weight，没有图的keras模型
            last_conv_layer_ind: 最后一个卷积层(activation之前)的index，可以为负数。如ResNet50中为-4 (add_16层)
            pred_layer_ind: 预测层（activation之前）的index，可以为负数。如ResNet50中为-1 (fc1000层)
            img_size: 模型处理的图片尺寸
        """
        self.class_dict = class_dict
        self.class_num = len(self.class_dict)
        self.img_size = img_size
        self.model = self.get_full_model(model_path)
        self.last_conv_layer_ind, self.pred_layer_ind = self.get_layer_indices()
        self.reduced_model = self.get_reduced_model()
        self.last_weight = self.get_last_weight()

    def get_full_model(self, model_path):
        """获取训练好的完整的模型
        """
        model_path_lower = model_path.lower()
        if 'resnet50' in model_path_lower:
            base_model = ResNet50(input_shape=self.img_size, weights=None, include_top=False)
        elif 'resnet101' in model_path_lower:
            classifier, _ = Classifiers.get('resnet101')
            base_model = classifier(input_shape=self.img_size, weights=None, include_top=False)
        elif 'seresnext50' in model_path_lower:
            classifier, _ = Classifiers.get('seresnext50')
            base_model = classifier(input_shape=self.img_size, weights=None, include_top=False)
        elif 'efficientnetb5' in model_path_lower:
            classifier, _ = Classifiers.get('efficientnetB5')
            base_model = classifier(input_shape=self.img_size, weights=None, include_top=False)
        else:
            raise ValueError('Model {:s} is not supported!'.format(model_path))
        model = add_new_last_layer(base_model, self.class_num)
        model.load_weights(model_path)
        print(model.summary())
        return model

    def get_layer_indices(self):
        """获取最后一层卷积层和预测层的layer index。
        注意：global_average_pooling和dense层可能会有多个，但这里只取最后一个。
        """
        last_conv_layer_ind, pred_layer_ind = 0, 0
        layer_cnt = 0
        for ind, layer in enumerate(self.model.layers):
            layer_cnt += 1
            if 'global_average_pooling' in layer.name.lower():
                last_conv_layer_ind = ind - 1
            if 'dense' in layer.name.lower():
                pred_layer_ind = ind
        print('最后一层卷积层为倒数第{:d}层，该层名字为{:s}.'.format((layer_cnt - last_conv_layer_ind),
                                                   self.model.layers[last_conv_layer_ind].name))
        print('预测层为倒数第{:d}层，该层名字为{:s}.'.format((layer_cnt - pred_layer_ind), self.model.layers[pred_layer_ind].name))
        return last_conv_layer_ind, pred_layer_ind

    def get_reduced_model(self):
        """
        抽取指定输入和输出的模型
        """
        ext_model = Model(inputs=self.model.input,
                          outputs=(self.model.layers[self.last_conv_layer_ind].output,
                                   self.model.layers[self.pred_layer_ind].output))
        return ext_model

    def preprocess_image(self, img):
        """图像预处理操作，先归一化再标准化。
        """
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        img = img / 255.0
        means = [0.65272259, 0.65272259, 0.65272259]  # 临时用该值，因为使用了新的resnet101，其他几个老模型都用下面都值
        stds = [0.08546339, 0.08546339, 0.0854634]
        # means = [0.4914009, 0.48215896, 0.4465308]
        # stds = [0.24703279, 0.24348423, 0.26158753]
        means = np.array(means)
        stds = np.array(stds)
        means = means.reshape(1, 1, 3)
        stds = stds.reshape(1, 1, 3)
        img -= means
        img /= (stds + 1e-6)
        img = np.expand_dims(img, axis=0)
        return img

    def get_last_weight(self):
        """
        提取最后一层的权重，如2048*1000
        """
        return self.model.layers[self.pred_layer_ind].get_weights()[0]

    def get_cam(self, img, class_id_list=None, merge=False):
        """
        获取图片在响应最大的类别的activation map, 或者指定类别上的activation map。
        Args：
            img: 原始输入图片
            class_id_list： 指定类别列表，为一个list. 不指定则取响应最大的类别。
            merge: 是否合并所有类别上的cam图, 注意无论class_id_list是否为None都可以执行此操作
        Return:
            output_dict: 一个字典，key为类别id，value为各个类别的activation map图，每张图为shape等于输入图片img的shape，只是通道为1。
                merge为True时，该字典增加一个键值对，key为‘merge’，value为各个类别叠加的图片。
        """
        ori_h, ori_w = img.shape[0:2]
        img = self.preprocess_image(img)
        # 获取最后一层卷积层和预测层
        last_conv_output, pred_vec = self.reduced_model.predict(img)
        # 4维变3维: 7 x 7 x 2048
        last_conv_output = np.squeeze(last_conv_output)
        h, w, c = last_conv_output.shape[0:3]
        pred_cls_ind = class_id_list

        output_dict = dict()
        # Case 1: 指定类别列表
        if pred_cls_ind is not None:
            # 获取各个类别的权重
            for _, cls_id in enumerate(pred_cls_ind):
                pred_cls_weights = self.last_weight[:, int(cls_id)]  # dim: (2048,)
                output_dict[cls_id] = np.dot(last_conv_output.reshape(
                    (h * w, c)), pred_cls_weights).reshape(h, w)  # dim: 224 x 224
        # Case 2: 未指定类别列表，取响应最大的类别
        else:
            # 获取响应最大的类别的index
            pred_cls_ind = np.argmax(pred_vec)
            # 获取响应最大的类别的权重
            pred_cls_weights = self.last_weight[:, pred_cls_ind]  # dim: (2048,)
            # 获取响应最大的类别的activation map
            output_dict[pred_cls_ind] = np.dot(last_conv_output.reshape(
                (h * w, c)), pred_cls_weights).reshape(h, w)  # dim: 224 x 224

        for out_cls, out_img in output_dict.items():
            # 将7*7的可视化feature map插值放大到原图大小（注意不是224*224）
            tmp_img = cv2.resize(out_img, (ori_w, ori_h))  # resize最大支持512维，我们的类别远少于这个。
            # 归一化
            tmp_img = np.maximum(tmp_img, 0)  # 把负响应的值去掉这一步很重要，否则可视化的背景有问题。
            x_min, x_max = np.min(tmp_img), np.max(tmp_img)
            tmp_img = ((tmp_img - x_min) / (x_max - x_min)) * 255.0
            tmp_img = tmp_img.astype(dtype=np.uint8)
            output_dict[out_cls] = tmp_img

        if merge:
            img_arr = np.zeros((ori_h, ori_w, len(output_dict.keys())))
            for i, (_, o_img) in enumerate(output_dict.items()):
                img_arr[:, :, i] = o_img
            # 合并预测出的所有类别上的activation map，保证每张图片输出的都是一张activation map
            merge_img = np.sum(img_arr, 2)
            merge_img = merge_img.astype(dtype=np.uint8)
            output_dict['merge'] = merge_img

        return output_dict

    def plot_cam(self, img_path, output_dir, class_id_list=None):
        """
        CAM效果展示：将原图和CAM叠加在一起。
        注意：plt用RGB格式来显示图片，需要先把bgr转为rgb
        Args:
            img_path: 原始输入图片路径
            output_dir: 结果图片保存目录
            class_id_list: 需要显示的类别列表，若不指定，则显示相应最大的类别。
        """
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        output_dict = self.get_cam(img, class_id_list, merge=True)
        cam_img = output_dict['merge']
        img = plt.imshow(cam_img, cmap='jet', alpha=0.3)  # 显示activation map
        plt.title('pred cls: {}'.format([self.class_dict[k] for k in output_dict.keys() if k != 'merge']))  # 打印类别信息
        plt.axis('off')
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        output_path = output_path.replace('jpg', 'png')  # savefig不支持jpg，要转为png
        plt.savefig(output_path)
        plt.close()


if __name__ == '__main__':
    from glob import glob
    from tqdm import tqdm

    class_dict = {0: 'ok', 1: '0', 2: '1', 3: '2', 4: '3', 5: '8'}
    # model_path = './model/efficientnetB5-ep008.h5' # -4, -2
    # model_path = './model/SeResnext50-ep017.h5' # -4, -2
    # model_path = './model/seresnext50_multilabel-ep050.h5' # -4, -2
    model_path = './input_file/set1_resnet101/resnet101.h5'  # -4, -2

    cam = CAM(class_dict, model_path, img_size=(224, 224, 3))

    src_dir = './input'
    des_dir = './output/' + os.path.basename(model_path).split('-')[0].lower()
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)

    # only plot cam in max class
    img_path_list = glob(os.path.join(src_dir, '*.jpg'))
    for img_path in tqdm(img_path_list):
        cam.plot_cam(img_path, des_dir)

    # #plot cam in specific classes
    # des_dir = './output'
    # img_path_list = glob(os.path.join(src_dir, '*.jpg'))
    # for img_path in tqdm(img_path_list):
    #     img = cv2.imread(img_path)
    #     cam.plot_cam(img_path, des_dir, [3])

    # # get pure cam in specific classes
    # des_dir = './output'
    # img_path_list = glob(os.path.join(src_dir, '*.jpg'))
    # for img_path in tqdm(img_path_list):
    #     img = cv2.imread(img_path)
    #     cam_img = cam.get_pure_cam(img, None, True)
    #     des_path = os.path.join(des_dir, os.path.basename(img_path))
    #     cv2.imwrite(des_path, cam_img)
