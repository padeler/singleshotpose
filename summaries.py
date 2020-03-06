import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from tensorboardX import SummaryWriter


box_lines = [
    [0, 4], [0, 2], [2, 6], [4, 6], # bottom 
    [1, 3], [1, 5], [3, 7], [5, 7], # top

    [6, 7], [2, 3], [0, 1], [4, 5], # sides
]

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

def _draw(overlay, anno, sc, thickness=1, palette=LABELS_256, show_bbox=True, line_pairs=None):
    boxes = anno['boxes'].cpu().numpy().copy()
    labels = anno['labels'].cpu().numpy()
    anchors = None
    if "anchors" in anno:
        anchors = anno['anchors'].cpu().numpy()
    else:
        anchors = None

    pose_boxes = None
    if "pose_boxes" in anno:
        pose_boxes = anno["pose_boxes"].cpu().numpy()
        
    for idx, (b, l) in enumerate(zip(boxes, labels)):
        b = (sc*b).astype(np.int32)
        pt1, pt2 = tuple(b[:2]), tuple(b[2:])
        
        clr = list(int(x) for x in palette[l%(len(palette))])
        if show_bbox:
            cv2.rectangle(overlay, pt1, pt2, clr, thickness)
            
        cv2.putText(overlay, "%d (%d)"%(idx, l), (pt1[0]+5, pt1[1]+20), 0, 0.5, clr, thickness)

        if pose_boxes is not None:
            bb = pose_boxes[idx]
            bb = (sc*bb).astype(np.int32)
            t1, t2 = tuple(bb[:2]), tuple(bb[2:])
            cv2.rectangle(overlay, t1, t2, clr, 3)
        
        if anchors is not None:
            obj_anchors = anchors[idx]
            for idx, p in enumerate(obj_anchors):
                p = p * sc
                c = list(int(v) for v in p)
                cv2.circle(overlay, tuple(c), 1, clr, thickness)
                cv2.putText(overlay, "%d"%idx, (c[0]+2, c[1]+2), 0, 0.3, (255, 255, 255), 1)
            
            if line_pairs is not None:
                for pair in line_pairs:
                    p1 = list(int(v) for v in sc * obj_anchors[pair[0]])
                    p2 = list(int(v) for v in sc * obj_anchors[pair[1]])
                    cv2.line(overlay, tuple(p1), tuple(p2), clr, thickness)


def process_image(image, anno, gt=None, img_size=256, show_bbox=True, show_3D=False):
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
        _draw(overlay, gt, sc, thickness=2, palette=[(0, 0, 0),], show_bbox=show_bbox, line_pairs=line_pairs)

    _draw(overlay, anno, sc, thickness=1, palette=LABELS_256, show_bbox=show_bbox, line_pairs=line_pairs)

    return overlay

    

def visualize_results(images, pred, gt=None, count=5, img_size=256, hstack=False, show_bbox=True, show_3D=False):
    '''
    Draw bounding boxes and labels on each image
    Scale all images to max_size (keep aspect ratio)
    finally stack them horizontally 
    '''
    m = count
    if gt is None:
        gt = [None,] * m

    viz_images = [process_image(im, p, g, img_size, show_bbox, show_3D) for im, p, g in zip(images[:m], pred[:m], gt[:m])]
    if hstack:
        return np.hstack(viz_images)
    else:
        return np.vstack(viz_images)



def create_summary(directory):
    writer = SummaryWriter(log_dir=os.path.join(directory))
    return writer



