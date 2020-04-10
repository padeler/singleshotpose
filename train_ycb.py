import datetime
import os
import time

import torch
import torch.utils.data

from saver import Saver
from train_tools import parse_args, create_logger, worker_init_fn

from summaries import create_summary
from torchvision import transforms
from engine import TrainEngine

# Singleshotpose stuff
import dataset
from cfg import parse_cfg
from utils import read_data_cfg
from darknet import Darknet
from region_loss import RegionLoss
import sys 

def main():

    args = parse_args()

    # set seed for repeatable results
    torch.manual_seed(args.seed)
    
    # Define Saver
    saver = Saver(args)
    saver.save_experiment_config()

    logger = create_logger(saver.experiment_dir, args.validate_only)

    net_cfg = parse_cfg(args.modelcfg)
    net_options   = net_cfg[0]
    
    batch_size    = int(net_options['batch'])
    learning_rate = float(net_options['learning_rate'])
    momentum      = float(net_options['momentum'])
    decay         = float(net_options['decay'])
    test_width  = int(net_options['test_width'])
    test_height = int(net_options['test_height'])
    init_width  = int(net_options['width'])
    init_height = int(net_options['height'])
    max_epochs    = int(net_options['max_epochs'])
    num_keypoints = int(net_options['num_keypoints'])
    steps         = [int(step) for step in net_options['steps'].split(',')]
    scales        = [float(scale) for scale in net_options['scales'].split(',')]
    bg_file_names = None # get_all_files('VOCdevkit/VOC2012/JPEGImages')

    args.test_width = test_width
    args.test_height = test_height

    # Data loading code
    logger.info("Loading train data from %s", args.experiment)
    trainset = dataset.listDataset(args.experiment, "train",
                                shape=(init_width, init_height),
                                transform=transforms.Compose([transforms.ToTensor(),]),
                                train=True,
                                seen=0,
                                batch_size=batch_size,
                                num_workers=args.workers,
                                bg_file_names=bg_file_names,
                                fixed_size=args.train_fixed_size)

    # # update net_options for the correct number of classes
    # net_cfg[-1]['classes'] = str(num_classes)
    # net_cfg[-2]['filters'] = str(18 + 1 + num_classes)

    model = Darknet(net_cfg)
    assert(model.num_anchors == 1)
    assert(len(model.anchors) == 0)


    region_loss = RegionLoss(num_keypoints=9, num_classes=model.num_classes, 
                            anchors=model.anchors, num_anchors=model.num_anchors, 
                            pretrain_num_epochs=args.pretrain_num_epochs)
    # Model settings
    epoch = 0


    model.print_network()
    model.seen = 0
    region_loss.iter  = model.iter
    region_loss.seen  = model.seen
    init_width        = model.width
    init_height       = model.height
    
    
    logger.info("Created model for %d classes and %d anchors. Weights init file %s", model.num_classes, model.num_anchors, args.weightfile)

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}


    logger.info("Creating train data loader")
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, **kwargs)



    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=learning_rate, momentum=momentum, weight_decay=decay)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1)


    if args.resume or args.validate_only:
        logger.info("Loading model weights from %s", args.weightfile)
        checkpoint = torch.load(args.weightfile, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # epoch = checkpoint.get('epoch', 0)
    else:
        model.load_weights_until_last(args.weightfile)

    
    model.to(device)
    
    logger.info("Creating tensorboard writer")
    writer = create_summary(saver.experiment_dir)

    # Save model graph to Tensorboard
    if args.add_graph:
        logger.info("Saving model graph to tensorboardx")
        try:
            dump_input = [torch.rand((3, init_height, init_width)).to(device),]
            writer.add_graph(model, (dump_input, ), verbose=False)
        except RuntimeError as e:
            logger.warning("Failed to save model graph. Message: %s", str(e), exc_info=False)


    train_engine = TrainEngine(args, device, logger, writer, saver)

    start_time = time.time()
    if not args.validate_only:
        logger.info("Start training. Total Epochs: %d", max_epochs)

        is_best = False

        min_error = epoch_error = 100000000
        while epoch < max_epochs:
            logger.info('[Epoch: %02d/%02d, numImages %5d, lr %f, best %.5f. Experiment: %s]', epoch, max_epochs, len(data_loader) * batch_size, optimizer.param_groups[0]['lr'], min_error, saver.experiment_dir)
            train_engine.train_one_epoch(model, region_loss, optimizer, data_loader, epoch)
            lr_scheduler.step()


            # evaluate (not on every epoch to save training time)
            if (epoch % 9 == 0 or epoch > max_epochs-10) and epoch >= args.pretrain_num_epochs:
                epoch_error = train_engine.evaluate(model, epoch)
                is_best = epoch_error < min_error
                logger.info("Evaluation score %0.4f, is_best=%d", epoch_error, is_best)
                if is_best:
                    min_error = epoch_error

            saver.save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'score': epoch_error,
                'args': args,
                'epoch': epoch,}, is_best)
            epoch += 1
    else:
        logger.info("Start evaluation.")
        score = train_engine.evaluate(model, 0)
        logger.info("Evaluation completed. Score: %f", score)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Total time %s', total_time_str)

    writer.close()


if __name__ == "__main__":
    main()
