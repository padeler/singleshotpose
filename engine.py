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
    def evaluate(self, model, epoch):
        # n_threads = torch.get_num_threads()
        # torch.set_num_threads(1)

        cpu_device = torch.device("cpu")
        model.eval()

        coco_evaluator.reset()


        i = test_loss = 0.0
        tbar = tqdm(data_loader, desc='\r', ascii=True, dynamic_ncols=True)
        
        for i, (image_cpu, targets_cpu) in enumerate(tbar):
            image = list(img.to(self.device) for img in image_cpu)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets_cpu]

            torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets_cpu, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            if i == 0:
                bgr = visualize_results(image_cpu, outputs, gt=targets_cpu, count=10, hstack=False)
                self.saver.save_image(bgr, "val", epoch)
                grid_image = torch.from_numpy(bgr[..., ::-1].transpose(2, 0, 1).copy())
                self.writer.add_image('val/grid', grid_image, epoch)

        # gather the stats from all processes
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        # torch.set_num_threads(n_threads)
        
        coco_evaluator.write_result(self.writer, self.logger, epoch)

        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.logger.info('Validation [Epoch: %d, numImages: %5d] Loss: %.3f', epoch, i * data_loader.batch_size + len(image), test_loss)

        bbox_ap_score = coco_evaluator.coco_eval['bbox'].stats[0]
        return coco_evaluator, bbox_ap_score

