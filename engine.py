import math
import sys
import time
import datetime
from collections import defaultdict, deque
from tqdm import tqdm

import torch

from summaries import visualize_results


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
            loss = region_loss(output, targets, epoch)



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
                self.logger.info("[%d/%d] Loss %0.4f (%0.4f) LR %0.5f Speed %0.3f samples/s mem %.0f", 
                                                i, num_img_tr, loss_value, train_loss/i, optimizer.param_groups[0]['lr'],
                                                (i*data_loader.batch_size)/(time.time()-st), (torch.cuda.max_memory_allocated() / 1024**2))


        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.logger.info('Epoch %d total loss: %.3f', epoch, train_loss)



    @torch.no_grad()
    def evaluate(self, model, test_loader, epoch):
        # n_threads = torch.get_num_threads()
        # torch.set_num_threads(1)
        def truths_length(truths):
            for i in range(50):
                if truths[i][1] == 0:
                    return i

        from torch.autograd import Variable # Useful info about autograd: http://pytorch.org/docs/master/notes/autograd.html
        from utils import get_region_boxes
        import numpy as np
        from utils import pnp
        im_width = 416
        im_height = 416

        
        cpu_device = torch.device("cpu")
        model.eval()

        use_cuda = True
        # Parameters
        num_keypoints = 9
        num_classes          = model.num_classes
        anchors              = model.anchors
        num_anchors          = model.num_anchors
        testtime             = True
        testing_error_trans  = 0.0
        testing_error_angle  = 0.0
        testing_error_pixel  = 0.0
        testing_samples      = 0.0
        errs_2d              = []
        errs_3d              = []
        errs_trans           = []
        errs_angle           = []
        errs_corner2D        = []
        self.logger.info("   Testing...")
        self.logger.info("   Number of test samples: %d" % len(test_loader.dataset))
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
                num_gts    = truths_length(truths)
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
                    R_gt, t_gt = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_gt, np.array(internal_calibration, dtype='float32'))
                    R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(internal_calibration, dtype='float32'))

                    # Compute errors
                    # Compute translation error
                    trans_dist   = np.sqrt(np.sum(np.square(t_gt - t_pr)))
                    errs_trans.append(trans_dist)

                    # Compute angle error
                    angle_dist   = calcAngularDistance(R_gt, R_pr)
                    errs_angle.append(angle_dist)

                    # Compute pixel error
                    Rt_gt        = np.concatenate((R_gt, t_gt), axis=1)
                    Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)
                    proj_2d_gt   = compute_projection(vertices, Rt_gt, internal_calibration) 
                    proj_2d_pred = compute_projection(vertices, Rt_pr, internal_calibration) 
                    norm         = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
                    pixel_dist   = np.mean(norm)
                    errs_2d.append(pixel_dist)

                    # Compute 3D distances
                    transform_3d_gt   = compute_transformation(vertices, Rt_gt) 
                    transform_3d_pred = compute_transformation(vertices, Rt_pr)  
                    norm3d            = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
                    vertex_dist       = np.mean(norm3d)    
                    errs_3d.append(vertex_dist)  

                    # Sum errors
                    testing_error_trans  += trans_dist
                    testing_error_angle  += angle_dist
                    testing_error_pixel  += pixel_dist
                    testing_samples      += 1

            t5 = time.time()

        # Compute 2D projection, 6D pose and 5cm5degree scores
        px_threshold = 5 # 5 pixel threshold for 2D reprojection error is standard in recent sota 6D object pose estimation works 
        eps          = 1e-5
        acc          = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
        acc3d        = len(np.where(np.array(errs_3d) <= vx_threshold)[0]) * 100. / (len(errs_3d)+eps)
        acc5cm5deg   = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
        corner_acc   = len(np.where(np.array(errs_corner2D) <= px_threshold)[0]) * 100. / (len(errs_corner2D)+eps)
        mean_err_2d  = np.mean(errs_2d)
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

        # Print test statistics
        self.logger.info("   Mean corner error is %f" % (mean_corner_err_2d))
        self.logger.info('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
        self.logger.info('   Acc using {} vx 3D Transformation = {:.2f}%'.format(vx_threshold, acc3d))
        self.logger.info('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
        self.logger.info('   Translation error: %f, angle error: %f' % (testing_error_trans/(nts+eps), testing_error_angle/(nts+eps)) )

        # Register losses and errors for saving later on
        # testing_iters.append(niter)
        # testing_errors_trans.append(testing_error_trans/(nts+eps))
        # testing_errors_angle.append(testing_error_angle/(nts+eps))
        # testing_errors_pixel.append(testing_error_pixel/(nts+eps))
        # testing_accuracies.append(acc)

