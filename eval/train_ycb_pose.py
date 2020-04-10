import os

def run():

    sets = [
            # "NbClass_06__TrainFrames_03_Frames",
            # "NbClass_06__TrainFrames_01_Percent",
            # "NbClass_06__TrainFrames_05_Percent",
            # "NbClass_06__TrainFrames_10_Percent",
            # "NbClass_06__TrainFrames_20_Percent",
            # "NbClass_06__TrainFrames_50_Percent",

            "NbClass_20__TrainFrames_03_Frames",
            "NbClass_20__TrainFrames_01_Percent",
            # "NbClass_20__TrainFrames_05_Percent",
            # "NbClass_20__TrainFrames_10_Percent",
            # "NbClass_20__TrainFrames_20_Percent",
            # "NbClass_20__TrainFrames_50_Percent",

            "NbClass_50__TrainFrames_03_Frames",
            "NbClass_50__TrainFrames_01_Percent",
            # "NbClass_50__TrainFrames_05_Percent",
            # "NbClass_50__TrainFrames_10_Percent",
            # "NbClass_50__TrainFrames_20_Percent",
            # "NbClass_50__TrainFrames_50_Percent",

            # "NbClass_92__TrainFrames_03_Frames",
            # "NbClass_92__TrainFrames_01_Percent",
            # "NbClass_92__TrainFrames_05_Percent",
            # "NbClass_92__TrainFrames_10_Percent",
            # "NbClass_92__TrainFrames_20_Percent",
            # "NbClass_92__TrainFrames_50_Percent",
            ]


    for s in sets:
        cmd = "python train_ycb.py --experiment ycb/experiments/%s --kw t2_%s --train-fixed-size 700 --valid-fixed-size 360 --weightfile cfg/darknet19_448.conv.23 --modelcfg cfg/yolo-pose.cfg"%(s, s)
        print("Training for set %s cmd: %s"%(s, cmd))
        os.system(cmd)



if __name__ == "__main__":
    run()
