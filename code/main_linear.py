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

def linear_regression(x, y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x ** 2)
    sumxy = sum(x * y)
    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
    return np.linalg.solve(A, b)


# the number of samples for one antenna
length = 16

# run rtls_aoa_iq_with_rtls_util_export_into_csv_log.py
# and write your path of the output folder of rtls_aoa_iq_with_rtls_util_export_into_csv_log.py here
filepath = r"D:\prog10\Desktop\桌面\210120\aoadata\rtls_agent\examples\rtls_aoa_iq_with_rtls_util_export_into_csv_log"
filepath = r"D:\prog10\Desktop\桌面\210120\rtls_agent\rtls_agent\examples\rtls_aoa_iq_with_rtls_util_export_into_csv_log"
filepath = r"..\data\2021_03_29_sunny_antenna2"
# put the address of the device we want to analyze (e.g. the address of PASSIVE)
# this address is used to filter data in the csv file
# when calculating AOA from PASSIVE, write the PASSIVE's address
# when calculating AOA from MASTER, write the MASTER's address
address = "80:6F:B0:EE:AC:E1"
for filename in [i for i in os.listdir(filepath) if '.csv' in i][17:]:
    results1 = []
    results2 = []
    try:
        truth = int(filename.split('.')[-2].split('_')[-1]) - 135
    except:
        truth = 180
    # if __name__ == '__main__':

    # open the newest csv file in the directory

    # filename = [i for i in os.listdir(filepath) if '.csv' in i][-2]
    #filename = r"D:\prog10\Desktop\桌面\210120\rtls_agent\rtls_agent\examples\rtls_aoa_iq_with_rtls_util_export_into_csv_log\03_16_2021_10_34_58_rtls_raw_iq_samples.csv"
    # #print the filename of the opened file. Please check the filename.

    df16 = []
    dfa1 = []
    with open(os.path.join(filepath, filename)) as f:

        # open file to write to
        #with open('out.csv', 'w') as g:

            # read csv
            xs = []  # list of I raw data
            ys = []  # list of Q raw data

            # skip the header of the csv file
            for line in f.readlines()[1:]:
                identifier, pkt, sample_idx, rssi, ant_array, channel, i, q = line.split(',')
                if identifier != address:
                    xs.append(int(i))
                    ys.append(int(q))

            # calculate phase angles from these I/Q in degrees: [-180, 180]
            phases_all = [iatan2sc(ys[i], xs[i]) * 180 / 128 for i in range(len(xs))]

    # diff added to phases_plus. phases_plus = phases_all + diff
            diff = 0

            # the last sample of the previous packet to calculate difference
            lstResult = 0

            # cut the data into packages
            package_size = 512

            # read from the 20st package
            # if read the 1th package, change the starting point of this range().
            #print(int(len(phases_all) / package_size))
            if(len(phases_all) < package_size*5): continue
            print(f'open csv file {filename} packetcnt {int(len(phases_all)/package_size)}')
            for pacID in range(0, int(len(phases_all) / package_size)):

                #print('-------------------reading the',pacID,'th packet---------------------------')
                # cut into packets
                phases_packet = phases_all[pacID * package_size: (pacID + 1) * package_size]

                # check packet head with the output of rtls_aoa_iq_with_rtls_util_export_into_csv_log.py
                #print("packet head: ", xs[pacID * package_size], ys[pacID * package_size])

                # remove the first 32 samples
                phases_packet = phases_packet[32:]

                # plot
                #plt.plot(phases_packet, 'b.', linewidth=0.5, markersize=3)
                #plt.title('the original phases')
                #plt.show()

                # calculate phases_plus
                # accumulate phase angles to remove the +179 -> -180 angle change\
                # result phases are saved in phases_plus: [-180, +inf]
                phases_plus = []

                # move difference across packets to 20
                diff -= phases_packet[0] - lstResult - 20

                # if sample drops over 180: compensate 360 to the rest of the samples
                # a switch of antennas only introduces no more than 90degrees
                # and phase between samples are about 20 degrees
                # there should not be more than 180 degrees difference between samples
                nextdiff = 0
                nextold = 0
                fixmem = []
                for j in range(len(phases_packet) - 1):
                    phases_plus.append(phases_packet[j] + diff)
                    nextdiff = phases_packet[j] + diff - nextold
                    nextold = phases_packet[j] + diff
                    olddiff = diff
                    temp = phases_packet[1 + j] - phases_packet[j] - 20
                    t2 = 0
                    if abs(temp) > 180:
                        t2 = - temp / abs(temp) * 360
                        assert abs(t2)==360
                        diff += t2
                        '''
                    temp = phases_packet[1 + j] - phases_packet[j] + t2 - 20
                    assert(abs(temp) <= 180)
                    ######debug!!!!
                    if abs(temp) > 90: ##fix threshold
                        # diff = olddiff - phases_packet[1 + j] + phases_packet[j] + nextdiff ## fix to last point
                        diff = olddiff - phases_packet[1 + j] + phases_packet[j] + 22 ## to 22
                        fixmem.append(j)'''


                # save the last packet for difference
                lstResult = phases_packet[-1]
                phases_plus.append(lstResult + diff)
                phases_diff = [phases_plus[j + 1] - phases_plus[j] for j in range(0, len(phases_plus) - 1)]

                ant_phase_diffs = [
                    [phases_diff[i] for i in range(len(phases_diff)) if int((i % (length * 3)) / length) == ant]
                    for ant in range(3)]
                ant_result_diffs = [0, 0, 0]
                for ant in range(3):
                    # plot the histogram of phase_diff of antenna in each packet
                    # plt.hist(ant_phase_diffs[ant], bins=40, density=True)

                    # calculate the histogram
                    (histogram, binplace) = np.histogram(ant_phase_diffs[ant], bins=40)

                    # find the position of the highest bin
                    highest_bin = binplace[int(np.argmax(np.array(histogram)))]

                    # make an average around this bin
                    # Parameter: how wide we want this average
                    average_range = (np.max(ant_phase_diffs[ant]) - np.min(ant_phase_diffs[ant])) / 4

                    # calculate the average
                    ant_result_diffs[ant] = np.average(
                        [k for k in ant_phase_diffs[ant] if abs(k - highest_bin) < average_range])
                    # print("ant_result_diff: ", ant_result_diffs[ant])
                avg_diff1 = np.average(ant_result_diffs)


                #phases_diff = [i if i > -100 else i + 360 for i in phases_diff]
                # plt.plot(phases_plus, 'b.', linewidth=0.5, markersize=0.5)
                # plt.title('the original phases with 360 plus')
                # plt.show()
                # calculate the difference between phases
                # we'll only use this phase_diff later

                # plt.subplot(2,2,1)
                # plt.plot(phases_diff, 'b.', linewidth=0.5, markersize=10)
                # plt.plot(np.array(fixmem), [ phases_diff[i] for i in fixmem],  'r.', linewidth=0.5, markersize=10)
                # plt.title('phases_plus diff1 phase[i+1] - phase[i]')
                # plt.subplot(2,2,2)
                # plt.plot([phases_diff[i] for i in range(len(phases_diff)) if int((i % (length * 3)) / length) == 0], 'b.', linewidth=0.5, markersize=10,label = "antenna0")
                # temp = [phases_diff[i] for i in range(len(phases_diff)) if i % 16 == 0 and int((i % (length * 3)) / length) == 0]
                # plt.plot(np.array(range(0,len(temp)))*16, temp, 'r.', linewidth=0.5, markersize=10)
                # plt.plot([0, len(temp)*16], [ant_result_diffs[0]  , ant_result_diffs[0]], 'b', linewidth=2, markersize=1)
                # plt.title('ant0')
                # plt.subplot(2,2,3)
                # plt.plot([phases_diff[i] for i in range(len(phases_diff)) if int((i % (length * 3)) / length) == 1], 'g.', linewidth=0.5, markersize=10,label = "antenna1")
                # temp = [phases_diff[i] for i in range(len(phases_diff)) if i % 16 == 0 and int((i % (length * 3)) / length) == 1]
                # plt.plot(np.array(range(0,len(temp)))*16, temp, 'r.', linewidth=0.5, markersize=10)
                # plt.plot([0, len(temp)*16], [ant_result_diffs[1]  , ant_result_diffs[1]], 'g', linewidth=2, markersize=1)
                # plt.title('ant1')
                # plt.subplot(2,2,4)
                # plt.plot([phases_diff[i] for i in range(len(phases_diff)) if int((i % (length * 3)) / length) == 2], 'r.', linewidth=0.5, markersize=10,label = "antenna2")
                # temp = [phases_diff[i] for i in range(len(phases_diff)) if i % 16 == 0 and int((i % (length * 3)) / length) == 2]
                # plt.plot(np.array(range(0,len(temp)))*16, temp, 'b.', linewidth=0.5, markersize=10)
                # plt.plot([0, len(temp)*16], [ant_result_diffs[2]  , ant_result_diffs[2]], 'r', linewidth=2, markersize=1)
                # plt.title('ant2')
                # plt.show()

                phases_diff_len = [phases_plus[j + length] - phases_plus[j] for j in
                                   range(0, len(phases_plus) - length)]
                # plt.plot(phases_diff_len)
                # plt.show()
                phases_diff_lens = [[phases_diff_len[i] for i in range(len(phases_diff_len)) if int((i % (length * 3)) / length) == ant]
                                    for ant in range(3)]
                phases_diff_lens_demo = [[phases_diff_len[i] for i in range(len(phases_diff_len)) if int((i % (length * 3)) / length) == ant and abs(i % length - 8) < 4]
                                    for ant in range(3)]
                plt.plot(phases_diff_lens_demo[0], 'b.', linewidth=0.5, markersize=3,label = "antenna0to1")
                plt.plot(phases_diff_lens_demo[1], 'g.', linewidth=0.5, markersize=3,label = "antenna1to2")
                plt.plot(phases_diff_lens_demo[2], 'r.', linewidth=0.5, markersize=3,label = "antenna2to0")
                col=['b','g','r']
                # for k in range(3):
                #     avg = np.average(phases_diff_lens_demo[k])
                #     plt.plot([0,len(phases_diff_lens_demo[k])],[avg,avg],col[k], linewidth=0.5, markersize=3)


                plt.title('phases_plus diff16 phase[i+length] - phase[i]')
                plt.legend()
                plt.show()
                # plt.plot(phases_diff_len, 'b.', linewidth=0.5, markersize=3)
                # col=['b','g','r']

                # calc index
                ant16avg0 = [np.average(phases_diff_lens_demo[i]) for i in range(3)]
                ant16avg = ant16avg0[:]
                ant16avg.sort()
                # print(ant16avg[0]-ant16avg[1] , ant16avg[1]-ant16avg[2])
                if (abs(ant16avg[0] - ant16avg[1]) > abs(ant16avg[1] - ant16avg[2])):
                    temp = ant16avg[0]
                    ant16avg[0] = ant16avg[2]
                    ant16avg[2] = temp
                ant16index = [[i for i in range(3) if ant16avg0[k] == ant16avg[i]][0] for k in range(3)]


                realdf1 = []
                for k in range(3):
                    avg = np.average(phases_diff_lens_demo[k])
                    # plt.plot([0,len(phases_diff_len)],[avg,avg],col[k], linewidth=0.5, markersize=3)
                    # plt.plot([0,len(phases_diff_len)],[ant_result_diffs[k]*length,ant_result_diffs[k]*length],col[k], linewidth=0.5, markersize=3)
                    temp = ant16avg0[k] + sin(truth*math.pi/180)*90
                    if(k==2):temp = ant16avg0[k] - sin(truth*math.pi/180)*90*2
                    realdf1.append(temp/length)

                    # plt.plot([0, len(phases_diff_len)], [temp, temp],
                    #      col[k], linewidth=0.5, markersize=3)
                # plt.plot([0,len(phases_diff_len)],[np.average(realdf1),np.average(realdf1)],'k', linewidth=0.5, markersize=3)
                #
                # plt.title('phases_plus diff16 (all antennas) phase[i+length] - phase[i]')
                # plt.show()

                # plt.subplot(2, 2, 1)
                # plt.plot(phases_diff, 'b.', linewidth=0.5, markersize=10)
                # plt.plot([0, len(phases_diff)], [np.average(realdf1), np.average(realdf1)], 'b', linewidth=2,
                #          markersize=1)
                # plt.title('phases_plus diff1 phase[i+1] - phase[i]')
                # plt.subplot(2, 2, 2)
                # plt.plot([phases_diff[i] for i in range(len(phases_diff)) if int((i % (length * 3)) / length) == 0],
                #          'b.', linewidth=0.5, markersize=10, label="antenna0")
                # temp = [phases_diff[i] for i in range(len(phases_diff)) if
                #         i % 16 == 0 and int((i % (length * 3)) / length) == 0]
                # plt.plot(np.array(range(0, len(temp))) * 16, temp, 'r.', linewidth=0.5, markersize=10)
                # plt.plot([0, len(temp) * 16], [realdf1[0], realdf1[0]], 'b', linewidth=2,
                #          markersize=1)
                # plt.plot([0, len(temp)*16], [ant_result_diffs[0]  , ant_result_diffs[0]], 'k', linewidth=2, markersize=1)
                # plt.title('ant0')
                # plt.subplot(2, 2, 3)
                # plt.plot([phases_diff[i] for i in range(len(phases_diff)) if int((i % (length * 3)) / length) == 1],
                #          'g.', linewidth=0.5, markersize=10, label="antenna1")
                # temp = [phases_diff[i] for i in range(len(phases_diff)) if
                #         i % 16 == 0 and int((i % (length * 3)) / length) == 1]
                # plt.plot(np.array(range(0, len(temp))) * 16, temp, 'r.', linewidth=0.5, markersize=10)
                # plt.plot([0, len(temp) * 16], [realdf1[1], realdf1[1]], 'g', linewidth=2,
                #          markersize=1)
                # plt.plot([0, len(temp)*16], [ant_result_diffs[1]  , ant_result_diffs[1]], 'k', linewidth=2, markersize=1)
                # plt.title('ant1')
                # plt.subplot(2, 2, 4)
                # plt.plot([phases_diff[i] for i in range(len(phases_diff)) if int((i % (length * 3)) / length) == 2],
                #          'r.', linewidth=0.5, markersize=10, label="antenna2")
                # temp = [phases_diff[i] for i in range(len(phases_diff)) if
                #         i % 16 == 0 and int((i % (length * 3)) / length) == 2]
                # plt.plot(np.array(range(0, len(temp))) * 16, temp, 'b.', linewidth=0.5, markersize=10)
                # plt.plot([0, len(temp) * 16], [realdf1[2], realdf1[2]], 'r', linewidth=2,
                #          markersize=1)
                # plt.plot([0, len(temp)*16], [ant_result_diffs[2]  , ant_result_diffs[2]], 'k', linewidth=2, markersize=1)
                # plt.title('ant2')
                # plt.show()
                #


                # print(ant16index)
                #ant16index: 01-2

                ant16index = [2,0,1]
                result = ((ant16avg0[ant16index[0]] + ant16avg0[ant16index[1]]) / 2 - ant16avg0[ant16index[2]]) / 3
                # print(phasetoangle(result))


                # plt.title('phases_plus poi 4~12 diff16 phase[i+length] - phase[i]:' + str(phasetoangle(result)))

                # plt.legend()
                # plt.show()
                # plt.subplot(1,2,2)
                # plt.plot(phases_diff_len, 'b.', linewidth=0.5, markersize=3)
                # plt.title('phases_plus diff16 (all antennas) phase[i+length] - phase[i]')
                # plt.show()

                # calculate the averages of each antenna for levelling

                # phase diff samples of each antenna
                # ant_phase_diffs = [[phasediff in ant1],[phasediff in ant2],[phasediff in ant3]]

                # find the most popular ant_phase_diffs of each antenna
                # then minus this phase_diff to spin the increasing lines to a flat position
                # in order to remove the influence of "different sample rates" of different antennas
                # (or whatever caused this error)

                #print("avgard", avg_diff1 * length)
                #plt.show()
                '''
                #modify phases
                s = 0
                phases_plus_modified = []
                for i in range(len(phases_plus)):
                    phases_plus_modified.append(phases_plus[i] - s)
                    s += ant_result_diffs[int((i % (length * 3)) / length)]
                #plt.plot(phases_plus_modified, 'r.', linewidth=0.5, markersize=3)
                #plt.title('phases - avg_phase_increase_of_this_antenna(around 20)')
                #plt.show()

                #validation
                phases_plus_modified_in_each_ant = [[0],[0],[0]]
                for i in range(int(len(phases_plus_modified) / length)):
                    ant = i % 3
                    temp = phases_plus_modified[i * length + 4] - phases_plus_modified_in_each_ant[ant][-1]
                    for j in range(4,12):
                        idx = i * length + j
                        phases_plus_modified_in_each_ant[ant].append(phases_plus_modified[idx] - temp)

                #plt.plot(phases_plus_modified_in_each_ant[0], 'r.', linewidth=0.5, markersize=3,label='phases_plus_modified_in_each_ant[0]')
                #plt.plot(phases_plus_modified_in_each_ant[1], 'g.', linewidth=0.5, markersize=3,label='phases_plus_modified_in_each_ant[1]')
                #plt.plot(phases_plus_modified_in_each_ant[2], 'b.', linewidth=0.5, markersize=3,label='phases_plus_modified_in_each_ant[2]')
                #plt.title('validation: the mid 8 points of each antenna should be flattened')
                #plt.legend()
                #plt.show()


                # phases_diff_modified = phases_diff minus the popular phase diffs of each antenna
                phase_diff_modified = [phases_plus_modified[j + length] - phases_plus_modified[j] for j in
                                   range(0, len(phases_plus_modified) - length)]
                phase_diff_modified_per_ant = [[phase_diff_modified[i] for i in range(len(phase_diff_modified)) if int((i % (length * 3)) / length) == ant] for ant in range(3)]
                # #print(phase_diff_modified_per_ant[0])
                ant_result_diffs16 = [0,0,0]
                for ant in range(3):

                    # plot the histogram of phase_diff of antenna in each packet
                    # plt.hist(phase_diff_modified_per_ant[ant], bins=40)
                    # calculate the histogram
                    (histogram, binplace) = np.histogram(phase_diff_modified_per_ant[ant], bins=40)

                    # find the position of the highest bin
                    highest_bin = binplace[int(np.argmax(np.array(histogram)))]

                    # make an average around this bin
                    # Parameter: how wide we want this average
                    average_range = (np.max(phase_diff_modified_per_ant[ant]) - np.min(phase_diff_modified_per_ant[ant])) / 8

                    print('highest bin',highest_bin,'now calc avg ranging from ',highest_bin - average_range, 'to', highest_bin + average_range)
                    # calculate the average
                    ant_result_diffs16[ant] = np.average(
                        [k for k in phase_diff_modified_per_ant[ant] if abs(k - highest_bin) < average_range])
                    print("ant_result_diff16: ", ant_result_diffs16[ant])
                    # plt.show()

                ant_result_diffs16.sort()
                if(abs(ant_result_diffs16[0] - ant_result_diffs16[1]) > (ant_result_diffs16[1] - ant_result_diffs16[2])):
                    temp = ant_result_diffs16[0]
                    ant_result_diffs16[0] = ant_result_diffs16[2]
                    ant_result_diffs16[2] = temp

                diff3 = -((ant_result_diffs16[0]+ant_result_diffs16[1])/2-ant_result_diffs16[2])/3

                #print('ant diffs[calculated by (updiff-downdiff)/3]:',phasetoangle(diff3))
                print(str(phasetoangle(diff3)))
                # results1.append(np.average(angles))
                results2.append(phasetoangle(diff3))

                '''
                # #plt.show()

                # plot phase_diff_modified
                # plt.plot(phase_diff_modified, 'r.', linewidth=0.5, markersize=3)
                # #plt.show()
                # plt.hist(phase_diff_modified, bins=400, density=True, alpha=0.7)
                # #plt.show()

                # find normal difflength
                # parameter: 24 for normal angles
                # samp = [j for j in phase_diff_modified if abs(j - 24) < 10]
                # diffconst = length * np.average(samp)
                # #print(np.average(samp), len(samp), len(phases_all), "diffconst with length", diffconst)

                # draw lines for the original phases_plus
                a1s = []
                for i in range(length, int(len(phases_plus)) - length, length):
                    # parameter: only use the mid length/2 points for linear regression
                    lenside = int(length / 4)

                    # do the linear_regression. the result line: y = a0 + a1 * j
                    a0, a1 = linear_regression(np.array(range(lenside, length - lenside)),
                                               np.array(phases_plus[i + lenside:i + length - lenside]))
                    a1s.append(a1)
                    plt.plot(np.array(range(i, i + length)), [a0 + a1 * (x - i) for x in range(i, i + length)], 'r.', linewidth=0.5, markersize=3)  # , label='phase calculated from I/Q output')
                    plt.plot(np.array(range(i, i + length)), phases_plus[i:i + length], 'b.', linewidth=0.5, markersize=3)
                plt.title("phases and lines - antenna original")
                plt.show()

                avg_diff_linear = np.average(a1s)

                # draw lines for phases_plus with antenna diff removed
                # phase_diff_modified_sum = [sum(phase_diff_modified[:i + 1]) for i in range(len(phase_diff_modified) - 1)]
                phases_plus_modified = [phases_plus[i] - i * avg_diff_linear for i in range(len(phases_plus))]
                lst = -10000
                lsta1 = -10000
                for i in range(length, int(len(phases_plus_modified)) - length, length):
                    # parameter: only use the mid length/2 points for linear regression
                    lenside = int(length / 4)

                    # do the linear_regression. the result line: y = a0 + a1 * j
                    a0, a1 = linear_regression(np.array(range(i+lenside, i+length - lenside)),
                                               np.array(phases_plus_modified[i + lenside:i + length - lenside]))
                    plt.plot(np.array(range(i, i + length)), [a0 + a1 * (x - i) for x in range(i, i + length)], 'r.', linewidth=0.5, markersize=3)  # , label='phase calculated from I/Q output')
                    plt.plot(np.array(range(i, i + length)), phases_plus_modified[i:i + length], 'b.', linewidth=0.5, markersize=3)
                    if(lst!=-10000):
                        df16.append(a0+a1*i-lst)
                        dfa1.append(abs(a1-lsta1))
                    lsta1 = a1
                    lst = a0+a1*i

                phases_modified_diff_len = [phases_plus_modified[j + length] - phases_plus_modified[j] for j in
                                            range(0, len(phases_plus_modified) - length)]
                #
                plt.plot(phases_modified_diff_len, 'b.', linewidth=0.5, markersize=3, label="modified phase")
                plt.plot([i - np.average(ant_result_diffs)*length for i in phases_diff_len], 'r.', linewidth=0.5, markersize=3, label="original phase")
                plt.title('phases_plus_modified diff across length phase[i+length] - phase[i]')
                plt.legend()
                plt.show()


    df162 = [(-df16[i]/2)  if i%3==1 else df16[i] for i in range(len(df16))]
    dfa2 = dfa1[:]
    dfa2.sort()

    threshold = 0.25
    df162_sel = [[df162[i] for i in range(len(df162)) if dfa1[i] < dfa2[int(len(dfa2)*threshold)] and i%3==k] for k in range(3)]

    for k in range(3):
        ans = np.average(df162_sel[k])
        print(ans,phasetoangle(ans),truth,ans/sin(truth/180*math.pi))
    # plt.plot([df16[i] for i in range(len(df16)) if i%3==0], [dfa1[i] for i in range(len(dfa1)) if i%3==0], 'r.', markersize=2, label = 'ant0-1')
    # plt.plot([(-df16[i]/2) for i in range(len(df16)) if i%3==1], [dfa1[i] for i in range(len(dfa1)) if i%3==1], 'g.', markersize=2, label = 'ant1-2')
    # plt.plot([df16[i] for i in range(len(df16)) if i%3==2], [dfa1[i] for i in range(len(dfa1)) if i%3==2], 'b.', markersize=2, label = 'ant2-0')
    # plt.show()

    '''
    from pyheatmap.heatmap import HeatMap
    data = []
    for i in range(len(df162)):
        tmp = [int(df162[i]*10), int(dfa1[i]*200), 1]
        data.append(tmp)
    heat = HeatMap(data)
    heat.heatmap(save_as="2"+filename+".png")  # 热图

    plt.xlabel('phasediff')
    plt.ylabel('gradientdiff')
    plt.title("linear regression-phasediff and gradientdiff "+str(truth))
    plt.legend()
    plt.show()
#plt.plot(results1, 'b.', linewidth=0.5, markersize=3, label="angle average")
    plt.plot(results2, 'r.', linewidth=0.5, markersize=3, label="diff / 3")
    # plt.plot(np.array([0,len(results1)]),np.array([np.average(results1),np.average(results1)]), 'b', linewidth=0.5, markersize=3, label="truth")
    plt.plot(np.array([0,len(results2)]),np.array([np.average(results2),np.average(results2)]), 'r', linewidth=0.5, markersize=3, label="truth")
    plt.plot(np.array([0,len(results1)]),np.array([truth,truth]), 'g', linewidth=0.5, markersize=3, label="truth")
    plt.title('results: '+filename)
    plt.savefig(filename[:-4])
    plt.show()
    '''


