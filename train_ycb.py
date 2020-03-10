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

def adjust_learning_rate(optimizer, epoch,  base_learning_rate, batch_size, steps, scales):
    lr = base_learning_rate
    for i in range(len(steps)):
        scale = scales[i]
        if epoch >= steps[i]:
            lr = lr * scale
        else:
            break
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    
    return lr



def main():

    args = parse_args()
    # Define Saver
    saver = Saver(args)
    saver.save_experiment_config()

    logger = create_logger(saver.experiment_dir, args.validate_only)

    # set seed for repeatable results
    torch.manual_seed(args.seed)
    

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
    steps         = [float(step) for step in net_options['steps'].split(',')]
    scales        = [float(scale) for scale in net_options['scales'].split(',')]
    bg_file_names = None # get_all_files('VOCdevkit/VOC2012/JPEGImages')




    # Data loading code
    logger.info("Loading train data from %s", args.experiment)
    trainset = dataset.listDataset(args.experiment, "train",
                                shape=(init_width, init_height),
                                shuffle=True,
                                transform=transforms.Compose([transforms.ToTensor(),]),
                                train=True,
                                seen=0,
                                batch_size=batch_size,
                                num_workers=args.workers,
                                bg_file_names=bg_file_names)

    logger.info("Loading test data from %s", args.experiment)    
    testset = dataset.listDataset(args.experiment, "valid",
                                shape=(test_width, test_height),
                                shuffle=False,
                                transform=transforms.Compose([transforms.ToTensor(),]), 
                                train=False)

    num_classes = 92
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
    
    model.load_weights_until_last(args.weightfile)
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


    logger.info("Creating test data loader")
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, **kwargs)

    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)


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

        epoch = 0
        is_best = False
        best_score = epoch_score = 0
        while epoch < max_epochs:
            logger.info('[Epoch: %02d/%02d, numImages %5d, lr %.5f, best %.5f. Experiment: %s]', epoch, max_epochs, len(data_loader) * batch_size, optimizer.param_groups[0]['lr'], best_score, saver.experiment_dir)
            train_engine.train_one_epoch(model, region_loss, optimizer, data_loader, epoch)
            new_lr = adjust_learning_rate(optimizer, epoch, learning_rate, batch_size, steps, scales)
            logger.info("LR: %0.6f", new_lr)

            # evaluate after every epoch
            if epoch%2 == 0 and epoch > 0:
                _, epoch_score = train_engine.evaluate(model, test_loader, epoch=epoch)
                is_best = epoch_score > best_score
                if is_best:
                    best_score = epoch_score

            saver.save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'score': epoch_score,
                'args': args,
                'epoch': epoch,}, is_best)
            epoch += 1
    else:
        logger.info("Start evaluation.")
        _, score = train_engine.evaluate(model, epoch=0)
        logger.info("Evaluation completed. Score: %f", score)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Total time %s', total_time_str)

    writer.close()


if __name__ == "__main__":
    main()