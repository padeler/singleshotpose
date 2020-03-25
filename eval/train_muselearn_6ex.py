import os

def run():

    sets = [
            "muselearn_pilot_set",
            ]


    for s in sets:
        cmd = "python train_ycb.py --experiment muselearn/experiments/%s --kw %s --weightfile cfg/darknet19_448.conv.23 --modelcfg cfg/yolo-pose.cfg"%(s, s)
        print("Training for set %s cmd: %s"%(s, cmd))
        os.system(cmd)



if __name__ == "__main__":
    run()
