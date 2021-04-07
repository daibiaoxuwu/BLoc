import math
import numpy as np
import os
import matplotlib.pyplot as plt
from AOA import *


def phasetoangle(phase):
    angle = float(phase / 90)
    angle = min(angle, 1)
    angle = max(angle, -1)
    return math.asin(angle) / math.pi * 180


length = 16

filepath = r"..\data\2021_03_29_sunny_antenna2"
address_master = "80:6F:B0:EE:AC:E1"
results1 = []
results2 = []
mem = []
metric_truth = []
metric_dist = []
metric_std = []

for filename in [i for i in os.listdir(filepath) if '.csv' in i][17:]:
    with open(os.path.join(filepath, filename)) as f:
        truth = int(filename.split('.')[-2].split('_')[-1]) - 135

        xs = []
        ys = []
        for line in f.readlines()[1:]:
            identifier, pkt, sample_idx, rssi, ant_array, channel, i, q = line.split(',')
            if identifier != address_master:
                xs.append(int(i))
                ys.append(int(q))
        phases_all = [iatan2sc(ys[i], xs[i]) * 180 / 128 for i in range(len(xs))]

        diff = 0
        lstResult = 0
        package_size = 512
        print(f'open csv file {filename} with {int(len(phases_all) / package_size)} packages')
        results0 = []
        for pacID in range(int(len(phases_all) / package_size)):
            phases_packet = phases_all[pacID * package_size: (pacID + 1) * package_size]
            phases_packet = phases_packet[32:]
            phases_plus = []
            diff -= phases_packet[0] - lstResult - 80
            nextdiff = 0
            nextold = 0
            fixmem = []
            for j in range(len(phases_packet) - 1):
                phases_plus.append(phases_packet[j] + diff)
                nextdiff = phases_packet[j] + diff - nextold
                nextold = phases_packet[j] + diff
                olddiff = diff
                temp = phases_packet[1 + j] - phases_packet[j] - 80
                t2 = 0
                if abs(temp) > 180:
                    t2 = - temp / abs(temp) * 360
                    assert abs(t2) == 360
                    diff += t2

            lstResult = phases_packet[-1]
            phases_plus.append(lstResult + diff)
            phases_diff = [phases_plus[j + 1] - phases_plus[j] for j in range(0, len(phases_plus) - 1)]
            # phases_diff = [i if i > -100 else i + 360 for i in phases_diff]
            plt.plot(phases_plus, 'b.', linewidth=0.5, markersize=0.5)
            plt.title('the original phases with 360 plus')
            plt.show()
            plt.plot(phases_diff, 'b', linewidth=0.5, markersize=2)
            plt.plot(np.array(fixmem), [ phases_diff[i] for i in fixmem],  'r.', linewidth=0.5, markersize=10)
            plt.title('phases_plus diff1 phase[i+1] - phase[i]')
            plt.show()
            '''
            plt.plot([phases_diff[i] for i in range(len(phases_diff)) if int((i % (length * 3)) / length) == 0], 'b.', linewidth=0.5, markersize=10,label = "antenna0")
            temp = [phases_diff[i] for i in range(len(phases_diff)) if i % 16 == 0 and int((i % (length * 3)) / length) == 0]
            plt.plot(np.array(range(0,len(temp)))*16, temp, 'r.', linewidth=0.5, markersize=10)
            plt.legend()
            plt.show()
            plt.plot([phases_diff[i] for i in range(len(phases_diff)) if int((i % (length * 3)) / length) == 1], 'g.', linewidth=0.5, markersize=10,label = "antenna1")
            temp = [phases_diff[i] for i in range(len(phases_diff)) if i % 16 == 0 and int((i % (length * 3)) / length) == 1]
            plt.plot(np.array(range(0,len(temp)))*16, temp, 'r.', linewidth=0.5, markersize=10)
            plt.legend()
            plt.show()
            plt.plot([phases_diff[i] for i in range(len(phases_diff)) if int((i % (length * 3)) / length) == 2], 'r.', linewidth=0.5, markersize=10,label = "antenna2")
            temp = [phases_diff[i] for i in range(len(phases_diff)) if i % 16 == 0 and int((i % (length * 3)) / length) == 2]
            plt.plot(np.array(range(0,len(temp)))*16, temp, 'b.', linewidth=0.5, markersize=10)
            plt.title('phases_plus diff1 phase[i+length] - phase[i]')
            plt.legend()
            plt.show()
            '''
            phases_diff_len = [phases_plus[j + length] - phases_plus[j] for j in
                               range(0, len(phases_plus) - length)]
            phases_diff_lens = [
                [phases_diff_len[i] for i in range(len(phases_diff_len)) if int((i % (length * 3)) / length) == ant]
                for ant in range(3)]
            phases_diff_lens_demo = [[phases_diff_len[i] for i in range(len(phases_diff_len)) if
                                      int((i % (length * 3)) / length) == ant and abs(i % length - 8) < 4]
                                     for ant in range(3)]
            plt.plot(phases_diff_lens_demo[0], 'b.', linewidth=0.5, markersize=3, label="antenna0to1")
            plt.plot(phases_diff_lens_demo[1], 'g.', linewidth=0.5, markersize=3, label="antenna1to2")
            plt.plot(phases_diff_lens_demo[2], 'r.', linewidth=0.5, markersize=3, label="antenna2to0")



            phases_diff_lens_demo_avg = [np.average(phases_diff_lens_demo[i]) for i in range(3)]
            angles = [phasetoangle(k) for k in phases_diff_lens_demo_avg]
            np.sort(angles)
            if (abs(phases_diff_lens_demo_avg[0] - phases_diff_lens_demo_avg[1]) > abs(
                    phases_diff_lens_demo_avg[1] - phases_diff_lens_demo_avg[2])):
                temp = phases_diff_lens_demo_avg[0]
                phases_diff_lens_demo_avg[0] = phases_diff_lens_demo_avg[2]
                phases_diff_lens_demo_avg[2] = temp

            # print("ant diffs[calculated by diff16-(avg_diff1)*16]", angles, 'avg', np.average(angles))
            # print(phases_diff_lens_demo_avg)
            result = ((phases_diff_lens_demo_avg[0] + phases_diff_lens_demo_avg[1]) / 2 - phases_diff_lens_demo_avg[
                2]) / 3
            print(phasetoangle(result))

            plt.plot([0, len(phases_diff_lens_demo[0])], [phases_diff_lens_demo_avg[0], phases_diff_lens_demo_avg[0]], 'b', linewidth=0.5, markersize=3, label="antenna0to1")
            plt.plot([0, len(phases_diff_lens_demo[0])], [phases_diff_lens_demo_avg[1], phases_diff_lens_demo_avg[1]], 'g', linewidth=0.5, markersize=3, label="antenna1to2")
            plt.plot([0, len(phases_diff_lens_demo[0])], [phases_diff_lens_demo_avg[2], phases_diff_lens_demo_avg[2]], 'r', linewidth=0.5, markersize=3, label="antenna2to0")
            plt.title(filename[-8:] + str(pacID) + ' : ' + str(int(phasetoangle(result))) + ' ' + str(int( phasetoangle((phases_diff_lens_demo_avg[0] - phases_diff_lens_demo_avg[2]) / 3))) + ' ' + str(int(phasetoangle((phases_diff_lens_demo_avg[1] - phases_diff_lens_demo_avg[2]) / 3))))
            plt.legend()
            plt.show()

            metric_truth.append(abs(phasetoangle(result) - truth))
            metric_dist.append(abs(phases_diff_lens_demo_avg[0] - phases_diff_lens_demo_avg[1]))
            metric_std.append(min(np.std(phases_diff_lens_demo[0]), np.std(phases_diff_lens_demo[1]),
                                  np.std(phases_diff_lens_demo[2])))

            results0.append(phasetoangle(result))

    print(results0)
    print(truth, np.average(results0))
    # plt.show()
    # plt.plot(results1, 'b.', linewidth=0.5, markersize=3, label="angle average")
    # plt.plot(results2, 'r.', linewidth=0.5, markersize=3, label="diff / 3")
    #
    # plt.plot(np.array([0,len(results1)]),np.array([truth,truth]), 'g', linewidth=0.5, markersize=3, label="truth")
    # plt.plot(np.array([0,len(results1)]),np.array([np.average(results1),np.average(results1)]), 'b', linewidth=0.2, markersize=3, label="avg angleavg")
    # plt.plot(np.array([0,len(results2)]),np.array([np.average(results2),np.average(results2)]), 'r', linewidth=0.2, markersize=3, label="avg dif/3")
    # plt.plot(np.array([0,len(results2)]),np.array([90,90]), 'k', linewidth=0.2, markersize=3, label="90")
    #
    # plt.title('results: '+filename)
    # plt.savefig(filename[:-4])
    # plt.clf()
    # plt.show()
    # print(truth,np.average(results1),np.average(results2))

plt.plot(metric_std, metric_truth, '.')
plt.xlabel('std')
plt.ylabel('truth')
plt.show()
