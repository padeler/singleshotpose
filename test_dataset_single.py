import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import numpy as np 
import cv2

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision import transforms

import dataset
import argparse
from utils import read_data_cfg

from cfg import parse_cfg
from darknet import Darknet

from utils import get_all_files, get_region_boxes

import matplotlib.pyplot as plt
import matplotlib

def get_cmap(labelCount, cmapName='jet', addbg=True):
    cmapGen = matplotlib.cm.get_cmap(cmapName, labelCount)
    cmap = cmapGen(np.arange(labelCount))
    cmap = cmap[:, 0:3]
    st0 = np.random.get_state()
    np.random.seed(42)
    perm = np.random.permutation(labelCount)
    np.random.set_state(st0)
    cmap = cmap[perm, :]

    # Add black color for 'background' class
    if addbg:
        cmap = np.vstack(((0.0, 0.0, 0.0), cmap))
    cmap = cmap[:labelCount] * 255
    return cmap.astype(np.uint8)

LABELS_256 = get_cmap(256, addbg=True)


box_lines = [
    [0, 4], [0, 2], [2, 6], [4, 6], # bottom 
    [1, 3], [1, 5], [3, 7], [5, 7], # top

    [6, 7], [2, 3], [0, 1], [4, 5], # sides
]


def _draw(overlay, anno, thickness=1, palette=LABELS_256, show_bbox=True, line_pairs=None):
    corners = anno[0, 1:-2].reshape(-1, 2) # XXX For single object annotation per image

    
    corners = (corners * overlay.shape[0]).astype(np.int32)

    # scale to img size
         
    for idx, c in enumerate(corners):
        c = tuple(c)
        
        clr = list(int(x) for x in palette[idx%(len(palette))])
            
        cv2.circle(overlay, tuple(c), 1, clr, thickness)
        cv2.putText(overlay, "%d"%idx, (c[0]+5, c[1]+20), 0, 0.5, clr, thickness)
            
    if line_pairs is not None:
        for pair in line_pairs:
            p1 = list(corners[1:][pair[0]])
            p2 = list(corners[1:][pair[1]])
            cv2.line(overlay, tuple(p1), tuple(p2), clr, thickness)


def process_image(image, gt=None, pred=None, img_size=256, show_3D=False):
    im = (image * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    h, w = im.shape[:2]
    big = h if h > w else w
    sc = img_size / big

    img_sc = cv2.resize(im, (0, 0), fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
    nh, nw = img_sc.shape[:2]

    overlay = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    overlay[:nh, :nw] = img_sc[..., ::-1]

    line_pairs = None
    if show_3D:
        line_pairs = box_lines

    if gt is not None:
        _draw(overlay, gt, thickness=2, palette=[(255, 255, 255),], line_pairs=line_pairs)

    if pred is not None:
        _draw(overlay, pred, thickness=2, palette=[(240, 40, 30),], line_pairs=line_pairs)

    return overlay

    

def visualize_results(images, gt=None, pred=None, count=5, img_size=256, hstack=False, show_3d=True):
    '''
    Draw bounding boxes and labels on each image
    Scale all images to max_size (keep aspect ratio)
    finally stack them horizontally 
    '''
    m = count
    if gt is None:
        gt = [None,] * m
    if pred is None:
        pred = [None,] * m

    viz_images = [process_image(im, g, p, img_size, show_3d) for im, g, p in zip(images[:m], gt[:m], pred[:m])]
    if hstack:
        return np.hstack(viz_images)
    else:
        return np.vstack(viz_images)



def run():

    logger = logging.getLogger()
    
        # Parse command window input
    parser = argparse.ArgumentParser(description='SingleShotPose')
    parser.add_argument('--datacfg', type=str, default='cfg/ape.data') # data config
    parser.add_argument('--modelcfg', type=str, default='cfg/yolo-pose.cfg') # network config
    parser.add_argument('--initweightfile', type=str, default='backup/init.weights') # initialization weights
    parser.add_argument('--pretrain_num_epochs', type=int, default=0) # how many epoch to pretrain
    args                = parser.parse_args()
    datacfg             = args.datacfg
    modelcfg            = args.modelcfg
    initweightfile      = args.initweightfile
    pretrain_num_epochs = args.pretrain_num_epochs

    print("ARGS: ", args)

    # Parse data configuration file
    data_options = read_data_cfg(datacfg)
    trainlist    = data_options['valid']
    gpus         = data_options['gpus']  
    num_workers  = int(data_options['num_workers'])
    backupdir    = data_options['backup']
    im_width     = int(data_options['width'])
    im_height    = int(data_options['height']) 
    fx           = float(data_options['fx'])
    fy           = float(data_options['fy'])
    u0           = float(data_options['u0'])
    v0           = float(data_options['v0'])

    print("DATA OPTIONS: ", data_options)

    # Parse network and training configuration parameters
    net_options   = parse_cfg(modelcfg)[0]
    loss_options  = parse_cfg(modelcfg)[-1]
    batch_size    = int(net_options['batch'])
    max_batches   = int(net_options['max_batches'])
    max_epochs    = int(net_options['max_epochs'])
    learning_rate = float(net_options['learning_rate'])
    momentum      = float(net_options['momentum'])
    decay         = float(net_options['decay'])
    conf_thresh   = float(net_options['conf_thresh'])
    num_keypoints = int(net_options['num_keypoints'])
    num_classes   = int(loss_options['classes'])
    num_anchors   = int(loss_options['num'])
    steps         = [float(step) for step in net_options['steps'].split(',')]
    scales        = [float(scale) for scale in net_options['scales'].split(',')]
    # anchors       = [float(anchor) for anchor in loss_options['anchors'].split(',')]

    print("NET OPTIONS: ", net_options)
    print("LOSS OPTIONS: ", loss_options)

    # Specifiy the model and the loss
    model       = Darknet(modelcfg)

    # # Model settings
    model.load_weights(initweightfile) 
    model.print_network()
    # model.seen        = 0
    # processed_batches = model.seen/batch_size
    init_width        = 416 # model.width
    init_height       = 416 # model.height
    batch_size = 1
    num_workers = 0

    # print("Size: ", init_width, init_height)

    bg_file_names = get_all_files('../VOCdevkit/VOC2012/JPEGImages')
    # Specify the number of workers
    use_cuda = True
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    
    logger.info("Loading data")

    # valid_dataset = dataset_multi.listDataset("../LINEMOD/duck/test_occlusion.txt", shape=(init_width, init_height),
    #                                             shuffle=False,
    #                                             objclass="duck",
    #                                             transform=transforms.Compose([
    #                                                 transforms.ToTensor(),
    #                                             ]))

    # Get the dataloader for training dataset

    dataloader = torch.utils.data.DataLoader(dataset.listDataset(trainlist, 
                                                                   shape=(init_width, init_height),
                                                            	   shuffle=False,
                                                            	   transform=transforms.Compose([transforms.ToTensor(),]), 
                                                            	   train=False,
                                                            	   seen=0,
                                                            	   batch_size=batch_size,
                                                            	   num_workers=num_workers, 
                                                                   bg_file_names=bg_file_names),
                                                batch_size=batch_size, shuffle=False, **kwargs)
    
    model.cuda()
    model.eval()
    
    delay = {True: 0, False: 1}
    paused = True

    # print("Classes in dataset ", num_classes)
    print("Batches in dataloader: ", len(dataloader))
    tbar = tqdm(dataloader, ascii=True, dynamic_ncols=True)
    for ii, s in enumerate(tbar):
        images, targets = s
        # print(ii, "IMAGES:" , images.shape)
        # print(ii, "TARGET\n", targets.shape)
        bs = images.shape[0]
        t = targets.cpu().numpy().reshape(bs, 50, -1)
        # print("TARGET [0, 0:1] \n", t[0, :1])
        # print("CLASSES ", t[0, :, 0])


        images_gpu = images.cuda()

        model_out = model(images_gpu).detach()
        all_boxes = np.array(get_region_boxes(model_out, num_classes, num_keypoints, anchor_dim=num_anchors)).reshape(batch_size, 1, -1)

        # print("Model OUT", all_boxes.shape)

        pred = np.zeros_like(all_boxes)
        pred[:, 0, 0] = all_boxes[:, 0, -1]
        pred[:, 0, 1:-2] = all_boxes[:, 0, :-3]

        viz = visualize_results(images, t, pred, img_size=416, show_3d=True)

        cv2.imshow("Res ", viz)

        k = cv2.waitKey(delay[paused])
        if k & 0xFF == ord('q'):
            break
        if k & 0xFF == ord('p'):
            paused = not paused


if __name__ == "__main__":
    run()
