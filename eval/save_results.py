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

from train_tools import parse_args
from test_dataset import visualize_results


from collections import OrderedDict
import json


def run():

    logger = logging.getLogger()
    args = parse_args()

    modelcfg            = args.modelcfg
    
    print("ARGS: ", args)

    # Parse network and training configuration parameters
    net_cfg  = parse_cfg(modelcfg)
    net_options   = net_cfg[0]
    loss_options  = net_cfg[-1]
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

    if not os.path.exists(args.weightfile):
        print("======================>Model weights for experiment %s not found"%args.experiment)
        return



    # Specifiy the model and the loss
    model = Darknet(net_cfg)
    logger.info("Loading model weights from %s", args.weightfile)
    checkpoint = torch.load(args.weightfile, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.print_network()
    model.cuda()
    model.eval()

    init_width        = model.width
    init_height       = model.height
    batch_size = 1
    num_workers = 4


    bg_file_names = None
    
    # Specify the number of workers
    use_cuda = True
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    
    logger.info("Loading data")

    ds = dataset.listDataset(args.experiment, "valid",
                        shape=(init_width, init_height),
                        shuffle=False,
                        transform=transforms.Compose([transforms.ToTensor(),]),
                        train=False,
                        seen=0,
                        batch_size=batch_size,
                        num_workers=num_workers, 
                        bg_file_names=bg_file_names)

    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, **kwargs)
    
    
    results = []
    delay = {True: 0, False: 1}
    paused = False

    # print("Classes in dataset ", num_classes)
    print("Batches in dataloader: ", len(dataloader))
    tbar = tqdm(dataloader, ascii=True, dynamic_ncols=True)
    for ii, s in enumerate(tbar):
        images, targets, meta = s
        # print(ii, "META:" , meta)
        # print(ii, "TARGET\n", targets.shape)
        bs = images.shape[0]
        t = targets.cpu().numpy().reshape(bs, 50, -1)
        # print("TARGET [0, 0:1] \n", t[0, :1])
        # print("CLASSES ", t[0, :, 0])

        images_gpu = images.cuda()

        model_out = model(images_gpu).detach()
        all_boxes = np.array(get_region_boxes(model_out, num_classes, num_keypoints)).reshape(batch_size, 1, -1)


        pred = np.zeros_like(all_boxes)
        pred[:, 0, 0] = all_boxes[:, 0, -1]
        pred[:, 0, 1:-2] = all_boxes[:, 0, :-3]


        # print("PRED", pred[0, 0])
        # print("GT", t[0, 0])
        img_path = meta["img_path"][0]
        obj_id = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        d = OrderedDict([
            ("OBJ", obj_id),
            ("IMG", img_path),
            ("PRED", pred[0, 0].tolist()),
            ("GT", t[0, 0].tolist()),
        ])
        results.append(d)

        
        # # Visualize:
        # img_paths = meta['img_path']
        # images = [cv2.imread(p) for p in img_paths]
        # viz = visualize_results(images, t, pred, img_size=640, show_3d=True)
        # cv2.imshow("Res ", viz)

        # k = cv2.waitKey(delay[paused])
        # if k & 0xFF == ord('q'):
        #     break
        # if k & 0xFF == ord('p'):
        #     paused = not paused

    

    header = OrderedDict([("experiment", args.experiment),
              ("weightfile", args.weightfile),
              ("num_images", len(dataloader)),
            ])

    res = OrderedDict( [
            ("header", header),
            ("results", results),
        ])

    dst_file = os.path.join(os.path.dirname(args.weightfile), "evaluation.json")
    print("Saving raw results in json format to:", dst_file)

    with open(dst_file, "w") as f:
        json.dump(res, f, indent=1)
    print("Done.")

if __name__ == "__main__":
    run()
