'''
Author: [egrt]
Date: 2022-08-23 11:44:15
LastEditors: [egrt]
LastEditTime: 2022-08-23 13:21:18
Description: HEAT的模型加载与预测
'''
import torch
import torch.nn as nn
from models.resnet import ResNetBackbone
from models.corner_models import HeatCorner
from models.edge_models import HeatEdge
from models.corner_to_edge import get_infer_edge_pairs
from datasets.data_utils import get_pixel_features
from PIL import Image
import numpy as np
import cv2
import scipy.ndimage.filters as filters
import skimage


class HEAT(object):
    #-----------------------------------------#
    #   注意修改model_path
    #-----------------------------------------#
    _defaults = {
        #-----------------------------------------------#
        #   backbone指向主干特征提取网络的地址
        #-----------------------------------------------#
        "backbone"            : 'model_data/G_FFHQ.pth',
        #-----------------------------------------------#
        #   corner_model指向焦点检测网络地址
        #-----------------------------------------------#
        "corner_model"        : 'model_data/G_FFHQ.pth',
        #-----------------------------------------------#
        #   edge_model指向边缘检测网络地址
        #-----------------------------------------------#
        "edge_model"          : 'model_data/G_FFHQ.pth',
        #-----------------------------------------------#
        #   image_shape模型预测图像的像素大小
        #-----------------------------------------------#
        "image_shape"       : [256, 256], 
        #-----------------------------------------------#
        #   corner_thresh为预测角点的阈值大小
        #-----------------------------------------------#
        "corner_thresh"     : 0.01,    
        #-----------------------------------------------#
        #   基于角点候选数的最大边数（不能大于6）
        #-----------------------------------------------#
        "corner_to_edge_multiplier": 3,
        #-----------------------------------------------#
        #   边缘推理筛选的迭代次数
        #-----------------------------------------------#
        "infer_times"       : 3,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : False,
    }

    #---------------------------------------------------#
    #   初始化MASKGAN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  
        self.generate()

    def generate(self):
        # 加载Backbone
        backbone = ResNetBackbone()
        strides = backbone.strides
        num_channels = backbone.num_channels
        backbone = nn.DataParallel(backbone)
        backbone = backbone.cuda()
        backbone.eval()
        # 加载角点检测模型
        corner_model = HeatCorner(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                                backbone_num_channels=num_channels)
        corner_model = nn.DataParallel(corner_model)
        corner_model = corner_model.cuda()
        corner_model.eval()
        # 加载边缘检测模型
        edge_model = HeatEdge(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                            backbone_num_channels=num_channels)
        edge_model = nn.DataParallel(edge_model)
        edge_model = edge_model.cuda()
        edge_model.eval()
        # 分别加载模型的地址
        self.backbone     = backbone.load_state_dict(self.backbone)
        self.corner_model = corner_model.load_state_dict(self.corner_model)
        self.edge_model   = edge_model.load_state_dict(self.edge_model)

    def detect_one_image(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        image       = image.resize(tuple(self.image_shape), Image.BICUBIC)
        # 将Image类转换为numpy
        image       = np.array(image, dtype=np.uint8)
        #   获取所有像素的位置编码, 默认的图像尺度为256
        pixels, pixel_features = get_pixel_features(image_size=self.image_shape[0])
        #   开始模型的预测
        with torch.no_grad():
            image_data = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                image_data = image_data.cuda()

            image_feats, feat_mask, all_image_feats = self.backbone(image)
            pixel_features = pixel_features.unsqueeze(0).repeat(image.shape[0], 1, 1, 1)
            preds_s1       = self.corner_model(image_feats, feat_mask, pixel_features, pixels, all_image_feats)

            c_outputs = preds_s1
            # 获取预测出的角点
            c_outputs_np = c_outputs[0].detach().cpu().numpy()
            # 筛选出大于阈值的角点的坐标
            pos_indices = np.where(c_outputs_np >= self.corner_thresh)
            pred_corners = pixels[pos_indices]
            # 获取对应预测角点的置信度
            pred_confs = c_outputs_np[pos_indices]
            # 根据预测角点的置信度进行非极大抑制
            pred_corners, pred_confs = corner_nms(pred_corners, pred_confs, image_size=c_outputs.shape[1])
            # 对角点两两排列组合，获取所有的角点对
            pred_corners, pred_confs, edge_coords, edge_mask, edge_ids = get_infer_edge_pairs(pred_corners, pred_confs)
            # 获取角点数量
            corner_nums = torch.tensor([len(pred_corners)]).to(image.device)
            max_candidates = torch.stack([corner_nums.max() * self.corner_to_edge_multiplier] * len(corner_nums), dim=0)
            # 无序不重复集合
            all_pos_ids = set()
            # 边缘置信度字典
            all_edge_confs = dict()
            # 推理的迭代次数为3次
            for tt in range(self.infer_times):
                if tt == 0:
                    # gt_values和边缘掩膜大小一样且初始值为0
                    gt_values = torch.zeros_like(edge_mask).long()
                    # 第一二维度的数值设置为2
                    gt_values[:, :] = 2

                # 开始预测边缘
                s1_logits, s2_logits_hb, s2_logits_rel, selected_ids, s2_mask, s2_gt_values = self.edge_model(image_feats, 
                    feat_mask,pixel_features,edge_coords, edge_mask,gt_values, corner_nums,max_candidates,True)
                num_total = s1_logits.shape[2]
                num_selected = selected_ids.shape[1]
                num_filtered = num_total - num_selected
                # 将输出值固定为(0,1)之间的概率分布
                s1_preds = s1_logits.squeeze().softmax(0)
                s2_preds_rel = s2_logits_rel.squeeze().softmax(0)
                s2_preds_hb = s2_logits_hb.squeeze().softmax(0)
                s1_preds_np = s1_preds[1, :].detach().cpu().numpy()
                s2_preds_rel_np = s2_preds_rel[1, :].detach().cpu().numpy()
                s2_preds_hb_np = s2_preds_hb[1, :].detach().cpu().numpy()

                selected_ids = selected_ids.squeeze().detach().cpu().numpy()
                # 进行筛选，将(0.9, 1)之间的设置为T，将(0.01,0.9)之间的设置为U,(0,0.01)之间的设置为F
                if tt != self.infer_times - 1:
                    s2_preds_np = s2_preds_hb_np

                    pos_edge_ids = np.where(s2_preds_np >= 0.9)
                    neg_edge_ids = np.where(s2_preds_np <= 0.01)
                    for pos_id in pos_edge_ids[0]:
                        actual_id = selected_ids[pos_id]
                        if gt_values[0, actual_id] != 2:
                            continue
                        all_pos_ids.add(actual_id)
                        all_edge_confs[actual_id] = s2_preds_np[pos_id]
                        gt_values[0, actual_id] = 1
                    for neg_id in neg_edge_ids[0]:
                        actual_id = selected_ids[neg_id]
                        if gt_values[0, actual_id] != 2:
                            continue
                        gt_values[0, actual_id] = 0
                    num_to_pred = (gt_values == 2).sum()
                    if num_to_pred <= num_filtered:
                        break
                else:
                    s2_preds_np = s2_preds_hb_np

                    pos_edge_ids = np.where(s2_preds_np >= 0.5)
                    for pos_id in pos_edge_ids[0]:
                        actual_id = selected_ids[pos_id]
                        if s2_mask[0][pos_id] is True or gt_values[0, actual_id] != 2:
                            continue
                        all_pos_ids.add(actual_id)
                        all_edge_confs[actual_id] = s2_preds_np[pos_id]
            pos_edge_ids = list(all_pos_ids)
            edge_confs = [all_edge_confs[idx] for idx in pos_edge_ids]
            pos_edges = edge_ids[pos_edge_ids].cpu().numpy()
            edge_confs = np.array(edge_confs)

            if self.image_size != 256:
                pred_corners = pred_corners / (self.image_size / 256)

        # return pred_corners, pred_confs, pos_edges, edge_confs, c_outputs_np
        #---------------------------------------------------------#
        #   此处开始推理结束
        #   开始在原图上根据角点坐标绘制角点与边缘
        #---------------------------------------------------------#
        if self.image_size != 256:
            pred_corners_viz = pred_corners * (self.image_size / 256)
        else:
            pred_corners_viz = pred_corners
        # 复制输入的原图
        viz_image    = image.copy()
        image_result = visualize_cond_generation(pred_corners_viz, pred_confs, viz_image)
        hr_image = Image.fromarray(np.uint8(image_result))
        return hr_image

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
#---------------------------------------------------------#
#   根据角点的置信度排序，并筛选出大于置信度的角点坐标
#---------------------------------------------------------#
def corner_nms(preds, confs, image_size):
    data = np.zeros([image_size, image_size])
    neighborhood_size = 5
    threshold = 0

    for i in range(len(preds)):
        data[preds[i, 1], preds[i, 0]] = confs[i]

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    results = np.where(maxima > 0)
    filtered_preds = np.stack([results[1], results[0]], axis=-1)

    new_confs = list()
    for i, pred in enumerate(filtered_preds):
        new_confs.append(data[pred[1], pred[0]])
    new_confs = np.array(new_confs)

    return filtered_preds, new_confs

def process_image(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = skimage.img_as_float(img)
    img = img.transpose((2, 0, 1))
    img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
    img = torch.Tensor(img).cuda()
    img = img.unsqueeze(0)
    return img
#---------------------------------------------------------#
#   将输入图像根据角点坐标进行可视化处理
#   不同于源代码，我们需要直接返回图像对象而不是保存到指定地址
#---------------------------------------------------------#
def visualize_cond_generation(positive_pixels, confs, image, gt_corners=None, prec=None, recall=None,
                              image_masks=None, edges=None, edge_confs=None):
    # 复制原图  
    image = image.copy()
    if confs is not None:
        viz_confs = confs

    if edges is not None:
        preds = positive_pixels.astype(int)
        c_degrees = dict()
        for edge_i, edge_pair in enumerate(edges):
            conf = (edge_confs[edge_i] * 2) - 1
            cv2.line(image, tuple(preds[edge_pair[0]]), tuple(preds[edge_pair[1]]), (255 * conf, 255 * conf, 0), 2)
            c_degrees[edge_pair[0]] = c_degrees.setdefault(edge_pair[0], 0) + 1
            c_degrees[edge_pair[1]] = c_degrees.setdefault(edge_pair[1], 0) + 1

    for idx, c in enumerate(positive_pixels):
        if edges is not None and idx not in c_degrees:
            continue
        if confs is None:
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 0, 255), -1)
        else:
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 0, 255 * viz_confs[idx]), -1)
        # if edges is not None:
        #    cv2.putText(image, '{}'.format(c_degrees[idx]), (int(c[0]), int(c[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
        #                0.5, (255, 0, 0), 1, cv2.LINE_AA)

    if gt_corners is not None:
        for c in gt_corners:
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 255, 0), -1)

    if image_masks is not None:
        mask_ids = np.where(image_masks == 1)[0]
        for mask_id in mask_ids:
            y_idx = mask_id // 64
            x_idx = (mask_id - y_idx * 64)
            x_coord = x_idx * 4
            y_coord = y_idx * 4
            cv2.rectangle(image, (x_coord, y_coord), (x_coord + 3, y_coord + 3), (127, 127, 0), thickness=-1)

    # if confs is not None:
    #    cv2.putText(image, 'max conf: {:.3f}'.format(confs.max()), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
    #                0.5, (255, 255, 0), 1, cv2.LINE_AA)
    if prec is not None:
        if isinstance(prec, tuple):
            cv2.putText(image, 'edge p={:.2f}, edge r={:.2f}'.format(prec[0], recall[0]), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'region p={:.2f}, region r={:.2f}'.format(prec[1], recall[1]), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(image, 'prec={:.2f}, recall={:.2f}'.format(prec, recall), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return image