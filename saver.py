import os
import shutil
import torch
from collections import OrderedDict
import glob
import json

import cv2

class Saver(object):

    def __init__(self, args):
        self.args = args
        if args.validate_only:
            root_dir='validate'
        else:
            root_dir = 'run'

        self.keyword = args.kw
        self.directory = os.path.join(root_dir, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, '%s_*'%self.keyword)))
        # print("Previous runs: ", self.runs)
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, '%s_%03d'%(self.keyword, run_id))
        self.vis_dir = os.path.join(self.experiment_dir, "vis")
        
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

        print("Experiment dir: %s"%self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['score']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))

            shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best.pth.tar'))
            

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.json')
        log_file = open(logfile, 'w')
        json.dump(vars(self.args), log_file, sort_keys=True, indent=4)
        log_file.close()


    def save_image(self, bgr, prefix, step_no):
        name = "%s_%08d.png" % (prefix, step_no)
        cv2.imwrite(self.vis_dir+os.sep+name, bgr)


