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
            # # "NbClass_50__TrainFrames_50_Percent",

            # "NbClass_92__TrainFrames_03_Frames",
            # "NbClass_92__TrainFrames_01_Percent",
            # "NbClass_92__TrainFrames_05_Percent",
            # "NbClass_92__TrainFrames_10_Percent",
            # "NbClass_92__TrainFrames_20_Percent",
            # # "NbClass_92__TrainFrames_50_Percent",
            ]


    for s in sets:
        cmd = "python eval/save_results.py --experiment ycb/experiments/%s --weightfile run/%s/t2_%s_000/model_best.pth.tar --modelcfg cfg/yolo-pose.cfg"%(s, s, s)
        print("Saving eval results for set %s cmd: \n%s"%(s, cmd))
        os.system(cmd)



if __name__ == "__main__":
    run()
