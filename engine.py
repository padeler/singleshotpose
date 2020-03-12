import math
import sys
import time
import datetime
from collections import defaultdict, deque
from tqdm import tqdm

import torch
from summaries import visualize_results

from torchvision import transforms
from torch.autograd import Variable  # Useful info about autograd: http://pytorch.org/docs/master/notes/autograd.html
from utils import get_region_boxes, calcAngularDistance, compute_transformation
import numpy as np
from utils import pnp
import dataset
from utils import read_data_cfg, get_3D_corners, get_camera_intrinsic, get_multi_region_boxes, fix_corner_order, corner_confidence, compute_projection
import glob
import os
from MeshPly import MeshPly


class TrainEngine(object):

    def __init__(self, args, device, logger, writer, saver):
        self.args = args
        self.device = device
        self.logger = logger
        self.writer = writer
        self.saver = saver

    def train_one_epoch(self, model, region_loss, optimizer, data_loader, epoch):
        model.train()
        num_img_tr = len(data_loader)
        st = time.time()

        train_loss = 0.0

        tbar = tqdm(data_loader, ascii=True, dynamic_ncols=True)
        for i, (images, targets) in enumerate(tbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            output = model(images)
            model.seen = model.seen + images.shape[0]
            region_loss.seen = model.seen

            # Compute loss, grow an array of losses for saving later on
            loss_dict = region_loss(output, targets, epoch)

            loss = loss_dict['x'] + loss_dict['y'] + loss_dict['cls']
            if epoch > self.args.pretrain_num_epochs:
                loss += loss_dict['conf']

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # self.logger.info("ITER %d, Anchor %0.5f  total %0.5f ", i, loss_dict['loss_pose_reg'].item(), loss_value)
            train_loss += loss_value
            global_step = i + num_img_tr * epoch

            self.writer.add_scalar('train/total_loss_iter', loss_value, global_step)

            # Show some inference results each epoch
            if i > 0 and i % self.args.log_freq == 0:
                self.logger.info("[%d/%d] Loss %0.4f (%0.4f) x %0.4f y %0.4f cls %0.4f conf %0.4f Speed %0.3f samples/s mem %.0f",
                                 i, num_img_tr, loss_value, train_loss/i,
                                 loss_dict['x'].item(), loss_dict['y'].item(), loss_dict['cls'].item(), loss_dict['conf'].item(),
                                 (i*data_loader.batch_size)/(time.time()-st), (torch.cuda.max_memory_allocated() / 1024**2))

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.logger.info('Epoch %d total loss: %.3f', epoch, train_loss)

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()

        if os.path.isdir(self.args.experiment):  # parse experiment dir
            all_data_files = glob.glob(self.args.experiment+os.sep+"**/*.data")
        else:
            all_data_files = [self.args.experiment, ]

        # for each sub-experiment data file run evaluation
        err = 0
        for exp in all_data_files[:1]:

            self.logger.info("Testing with data from %s", exp)
            _, exp_err = self._evaluate_one_object(exp, model)
            err += exp_err

        return err/len(all_data_files)

    def _evaluate_one_object(self, datacfg, model):

        from train import test
        # Parse configuration files
        options = read_data_cfg(datacfg)
        meshname = options['mesh']
        fx = float(options['fx'])
        fy = float(options['fy'])
        u0 = float(options['u0'])
        v0 = float(options['v0'])
        vx_threshold = float(options['diam']) * 0.1  # threshold for the ADD metric

        # Read object model information, get 3D bounding box corners
        mesh = MeshPly(meshname)
        vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
        corners3D = get_3D_corners(vertices)
        num_keypoints = 9
        num_labels = num_keypoints*2+3  # + 2 for image width, height, +1 for image class

        im_width = model.width
        im_height = model.height

        # Read intrinsic camera parameters
        internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)

        testset = dataset.listDataset(datacfg, "valid",
                                      shape=(im_width, im_height),
                                      shuffle=False,
                                      transform=transforms.Compose([transforms.ToTensor(), ]),
                                      train=False)

        kwargs = {'num_workers': 4, 'pin_memory': True}
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, **kwargs)

        self.logger.info("Testing. Number of test samples: %d", len(test_loader.dataset))
        acc, mean_px_err = self.__eval_tekin(model, test_loader, vertices, corners3D, internal_calibration, vx_threshold)
        return acc, mean_px_err

    def __eval_tekin(self, model, test_loader, vertices, corners3D, internal_calibration, vx_threshold):
        def truths_length(truths):
            for i in range(50):
                if truths[i][1] == 0:
                    return i

        # Set the module in evaluation mode (turn off dropout, batch normalization etc.)
        model.eval()
        # Parameters
        num_keypoints = model.num_keypoints
        im_width = model.width
        im_height = model.height
        use_cuda = True
        num_classes = model.num_classes
        anchors = model.anchors
        num_anchors = model.num_anchors
        testtime = False
        testing_error_trans = 0.0
        testing_error_angle = 0.0
        testing_error_pixel = 0.0
        testing_samples = 0.0
        errs_2d = []
        errs_3d = []
        errs_trans = []
        errs_angle = []
        errs_corner2D = []

        notpredicted = 0
        # Iterate through test examples
        for batch_idx, (data, target) in enumerate(test_loader):
            t1 = time.time()
            # Pass the data to GPU
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
            data = Variable(data, volatile=True)
            t2 = time.time()
            # Formward pass
            output = model(data).data
            t3 = time.time()
            # Using confidence threshold, eliminate low-confidence predictions
            all_boxes = get_region_boxes(output, num_classes, num_keypoints)
            t4 = time.time()
            # Iterate through all batch elements
            for box_pr, target in zip([all_boxes], [target[0]]):
                # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
                truths = target.view(-1, num_keypoints*2+3)
                # Get how many objects are present in the scene
                num_gts = truths_length(truths)
                # Iterate through each ground-truth object
                for k in range(num_gts):
                    box_gt = list()
                    for j in range(1, 2*num_keypoints+1):
                        box_gt.append(truths[k][j])
                    box_gt.extend([1.0, 1.0])
                    box_gt.append(truths[k][0])

                    # Denormalize the corner predictions
                    corners2D_gt = np.array(np.reshape(box_gt[:num_keypoints*2], [num_keypoints, 2]), dtype='float32')
                    corners2D_pr = np.array(np.reshape(box_pr[:num_keypoints*2], [num_keypoints, 2]), dtype='float32')
                    corners2D_gt[:, 0] = corners2D_gt[:, 0] * im_width
                    corners2D_gt[:, 1] = corners2D_gt[:, 1] * im_height
                    corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
                    corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height

                    # Compute corner prediction error
                    corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
                    corner_dist = np.mean(corner_norm)
                    errs_corner2D.append(corner_dist)

                    # Compute [R|t] by pnp
                    R_gt, t_gt = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)),
                                              dtype='float32'),  corners2D_gt, np.array(internal_calibration, dtype='float32'))
                    R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)),
                                              dtype='float32'),  corners2D_pr, np.array(internal_calibration, dtype='float32'))

                    # Compute errors
                    # Compute translation error
                    trans_dist = np.sqrt(np.sum(np.square(t_gt - t_pr)))
                    errs_trans.append(trans_dist)

                    # Compute angle error
                    angle_dist = calcAngularDistance(R_gt, R_pr)
                    errs_angle.append(angle_dist)

                    # Compute pixel error
                    Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
                    Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
                    proj_2d_gt = compute_projection(vertices, Rt_gt, internal_calibration)
                    proj_2d_pred = compute_projection(vertices, Rt_pr, internal_calibration)
                    norm = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
                    pixel_dist = np.mean(norm)
                    errs_2d.append(pixel_dist)

                    # Compute 3D distances
                    transform_3d_gt = compute_transformation(vertices, Rt_gt)
                    transform_3d_pred = compute_transformation(vertices, Rt_pr)
                    norm3d = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
                    vertex_dist = np.mean(norm3d)
                    errs_3d.append(vertex_dist)

                    # Sum errors
                    testing_error_trans += trans_dist
                    testing_error_angle += angle_dist
                    testing_error_pixel += pixel_dist
                    testing_samples += 1

            t5 = time.time()

        # Compute 2D projection, 6D pose and 5cm5degree scores
        px_threshold = 5  # 5 pixel threshold for 2D reprojection error is standard in recent sota 6D object pose estimation works
        eps = 1e-5
        acc = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
        acc3d = len(np.where(np.array(errs_3d) <= vx_threshold)[0]) * 100. / (len(errs_3d)+eps)
        acc5cm5deg = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
        corner_acc = len(np.where(np.array(errs_corner2D) <= px_threshold)[0]) * 100. / (len(errs_corner2D)+eps)
        mean_err_2d = np.mean(errs_2d)
        mean_corner_err_2d = np.mean(errs_corner2D)
        nts = float(testing_samples)

        if testtime:
            print('-----------------------------------')
            print('  tensor to cuda : %f' % (t2 - t1))
            print('         predict : %f' % (t3 - t2))
            print('get_region_boxes : %f' % (t4 - t3))
            print('            eval : %f' % (t5 - t4))
            print('           total : %f' % (t5 - t1))
            print('-----------------------------------')

        trans_err = testing_error_trans/(nts+eps)
        angle_err = testing_error_angle/(nts+eps)
        pixel_err = testing_error_pixel/(nts+eps)

        # Print test statistics
        self.logger.info("   Mean corner error is %f" % (mean_corner_err_2d))
        self.logger.info('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
        self.logger.info('   Acc using {} vx 3D Transformation = {:.2f}%'.format(vx_threshold, acc3d))
        self.logger.info('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
        self.logger.info('   Translation error: %f, angle error: %f' % (trans_err, angle_err))

        # Register losses and errors for saving later on
        return acc, mean_corner_err_2d
