# showing COOP rouge resuls by steps
import json
import os
import sys


def get_steps(x):
    return int(x.split('_', 2)[1].replace(".json", ""))


datadir = sys.argv[1]
files = os.listdir(datadir + '/')
files = list(filter(lambda x: x.startswith("metrics-step_"), files))
files.sort(key=get_steps)  # sort by step

print("                :   %-12s %-12s %-12s" % ("R1", "R2", "RL"))
for fn in files:
    with open(datadir + "/" + fn, "r") as f:
        steps = get_steps(fn)
        obj = json.loads(f.read())
        print("Steps: %-7i  :   %-12f %-12f %-12f" % (steps,
                                                   obj['dev']['rouge-1_sum_f'] * 100,
                                                   obj['dev']['rouge-2_sum_f'] * 100,
                                                   obj['dev']['rouge-l_sum_f'] * 100))
