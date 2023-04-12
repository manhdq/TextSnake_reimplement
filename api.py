import os
import time
import cv2
import numpy as np
import torch
import pprint
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from dataset.deploy import DeployDataset
from network.textnet import TextNet
from util.detection import TextDetector
from util.augmentation import BaseTransform
from util.config import config as cfg, update_config, print_config
from util.option import BaseOptions
from util.visualize import visualize_detection
from util.misc import to_device, mkdirs, rescale_result


def inference(detector, test_loader):
    
    total_time = 0.
    contours_list = []

    for i, (image, meta) in enumerate(test_loader):

        image = to_device(image)


        idx = 0 # test mode can only run with batch_size == 1

        # get detection result
        contours, output = detector.detect(image)

        # visualization
        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
        img_show, contours = rescale_result(img_show, contours, H, W)
        contours_list.append(contours)

    return contours_list


def main(imgs):
    if isinstance(imgs, str):
        imgs = [imgs]

    
    testset = DeployDataset(
        images=imgs,
        transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
    )
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = TextNet(is_training=False, backbone=cfg.net)
    model_path = "weights/textsnake_vgg_180.pth"
    # model_path = os.path.join(cfg.save_dir, cfg.exp_name, \
    #           'textsnake_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
    cfg.device='cpu'
    model.load_model(model_path, device=cfg.device)

    # copy to cuda
    model = model.to(cfg.device)
    # if cfg.cuda:
    #     cudnn.benchmark = True
    detector = TextDetector(model, tr_thresh=cfg.tr_thresh, tcl_thresh=cfg.tcl_thresh)

    contours_list = inference(detector, test_loader)
    bboxes_list = []
    for contours in contours_list:
        bboxes = []
        for contour in contours:
            x1 = contour[:, 0].min()
            y1 = contour[:, 1].min()
            x2 = contour[:, 0].max()
            y2 = contour[:, 1].max()
            bboxes.append([x1, y1, x2, y2])
        bboxes_list.append(bboxes)

    return bboxes_list


if __name__ == "__main__":

    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)

    vis_dir = os.path.join(cfg.vis_dir, '{}_deploy'.format(cfg.exp_name))
    if not os.path.exists(vis_dir):
        mkdirs(vis_dir)

    imgs = ['data/total-text/Images/Test/img5.jpg',
            'data/total-text/Images/Test/img6.jpg',
            'data/total-text/Images/Test/img7.jpg']
    results = main(imgs)