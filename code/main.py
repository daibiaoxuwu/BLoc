import math
import numpy as np
import os
import matplotlib.pyplot as plt
from AOA import *


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

# put the address of the device we want to analyze (e.g. the address of PASSIVE)
# this address is used to filter data in the csv file
# when calculating AOA from PASSIVE, write the PASSIVE's address
# when calculating AOA from MASTER, write the MASTER's address
address = "80:6F:B0:EE:AC:E1"

if __name__ == '__main__':

    # open the newest csv file in the directory

    filename = [i for i in os.listdir(filepath) if '.csv' in i][-2]
    #filename = r"D:\prog10\Desktop\桌面\210120\rtls_agent\rtls_agent\examples\rtls_aoa_iq_with_rtls_util_export_into_csv_log\03_16_2021_10_34_58_rtls_raw_iq_samples.csv"
    # print the filename of the opened file. Please check the filename.
    print(f'open csv file {filename}')

    with open(os.path.join(filepath, filename)) as f:

        # open file to write to
        with open('out.csv', 'w') as g:

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
            print(int(len(phases_all) / package_size))
            for pacID in range(0, int(len(phases_all) / package_size)):

                print('-------------------reading the',pacID,'th packet---------------------------')
                # cut into packets
                phases_packet = phases_all[pacID * package_size: (pacID + 1) * package_size]

                # check packet head with the output of rtls_aoa_iq_with_rtls_util_export_into_csv_log.py
                print("packet head: ", xs[pacID * package_size], ys[pacID * package_size])

                # remove the first 32 samples
                phases_packet = phases_packet[32:]

                # plot
                plt.plot(phases_packet, 'b.', linewidth=0.5, markersize=3)
                plt.title('the original phases')
                plt.show()

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
                for j in range(len(phases_packet) - 1):
                    phases_plus.append(phases_packet[j] + diff)
                    temp = phases_packet[1 + j] - phases_packet[j] - 20
                    if abs(temp) > 180:
                        t2 = - temp / abs(temp) * 360
                        diff += t2

                # save the last packet for difference
                lstResult = phases_packet[-1]
                phases_plus.append(lstResult + diff)

                plt.plot(phases_plus, 'b.', linewidth=0.5, markersize=0.5)
                plt.title('the original phases with 360 plus')
                plt.show()
                # calculate the difference between phases
                # we'll only use this phase_diff later
                phases_diff = [phases_plus[j + 1] - phases_plus[j] for j in range(0, len(phases_plus) - 1)]
                #phases_diff = [i if i > -100 else i + 360 for i in phases_diff]

                plt.plot(phases_diff, 'b.', linewidth=0.5, markersize=3)
                plt.title('phases_plus diff1 phase[i+1] - phase[i]')
                plt.show()
                plt.plot([phases_diff[i] for i in range(len(phases_diff)) if int((i % (length * 3)) / length) == 0], 'b.', linewidth=0.5, markersize=3,label = "antenna0")
                plt.legend()
                plt.show()
                plt.plot([phases_diff[i] for i in range(len(phases_diff)) if int((i % (length * 3)) / length) == 1], 'g.', linewidth=0.5, markersize=3,label = "antenna1")
                plt.legend()
                plt.show()
                plt.plot([phases_diff[i] for i in range(len(phases_diff)) if int((i % (length * 3)) / length) == 2], 'r.', linewidth=0.5, markersize=3,label = "antenna2")
                plt.title('phases_plus diff1 phase[i+length] - phase[i]')
                plt.legend()
                plt.show()

                phases_diff_len = [phases_plus[j + length] - phases_plus[j] for j in
                                   range(0, len(phases_plus) - length)]
                phases_diff_lens = [[phases_diff_len[i] for i in range(len(phases_diff_len)) if int((i % (length * 3)) / length) == ant]
                                    for ant in range(3)]

                plt.plot(phases_diff_lens[0], 'b.', linewidth=0.5, markersize=3,label = "antenna0to1")
                plt.plot(phases_diff_lens[1], 'g.', linewidth=0.5, markersize=3,label = "antenna1to2")
                plt.plot(phases_diff_lens[2], 'r.', linewidth=0.5, markersize=3,label = "antenna2to0")
                plt.title('phases_plus diff16 phase[i+length] - phase[i]')
                plt.legend()
                plt.show()
                plt.plot(phases_diff_len, 'b.', linewidth=0.5, markersize=3)
                plt.title('phases_plus diff16 (all antennas) phase[i+length] - phase[i]')
                plt.show()

                # calculate the averages of each antenna for levelling

                # phase diff samples of each antenna
                # ant_phase_diffs = [[phasediff in ant1],[phasediff in ant2],[phasediff in ant3]]
                ant_phase_diffs = [
                    [phases_diff[i] for i in range(len(phases_diff)) if int((i % (length * 3)) / length) == ant]

                    for ant in range(3)]

                # find the most popular ant_phase_diffs of each antenna
                # then minus this phase_diff to spin the increasing lines to a flat position
                # in order to remove the influence of "different sample rates" of different antennas
                # (or whatever caused this error)
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
                    print("ant_result_diff: ", ant_result_diffs[ant])
                avg_diff1 = np.average(ant_result_diffs)
                print("avgard", avg_diff1 * length)
                # plt.show()

                #modify phases
                s = 0
                phases_plus_modified = []
                for i in range(len(phases_plus)):
                    phases_plus_modified.append(phases_plus[i] - s)
                    s += ant_result_diffs[int((i % (length * 3)) / length)]
                plt.plot(phases_plus_modified, 'r.', linewidth=0.5, markersize=3)
                plt.title('phases - avg_phase_increase_of_this_antenna(around 20)')
                plt.show()

                #validation
                phases_plus_modified_in_each_ant = [[0],[0],[0]]
                for i in range(int(len(phases_plus_modified) / length)):
                    ant = i % 3
                    temp = phases_plus_modified[i * length + 4] - phases_plus_modified_in_each_ant[ant][-1]
                    for j in range(4,12):
                        idx = i * length + j
                        phases_plus_modified_in_each_ant[ant].append(phases_plus_modified[idx] - temp)

                plt.plot(phases_plus_modified_in_each_ant[0], 'r.', linewidth=0.5, markersize=3,label='phases_plus_modified_in_each_ant[0]')
                plt.plot(phases_plus_modified_in_each_ant[1], 'g.', linewidth=0.5, markersize=3,label='phases_plus_modified_in_each_ant[1]')
                plt.plot(phases_plus_modified_in_each_ant[2], 'b.', linewidth=0.5, markersize=3,label='phases_plus_modified_in_each_ant[2]')
                plt.title('validation: the mid 8 points of each antenna should be flattened')
                plt.legend()
                plt.show()


                # phases_diff_modified = phases_diff minus the popular phase diffs of each antenna
                phase_diff_modified = [phases_plus_modified[j + length] - phases_plus_modified[j] for j in
                                   range(0, len(phases_plus_modified) - length)]
                phase_diff_modified_per_ant = [[phase_diff_modified[i] for i in range(len(phase_diff_modified)) if int((i % (length * 3)) / length) == ant] for ant in range(3)]
                # print(phase_diff_modified_per_ant[0])
                ant_result_diffs16 = [0,0,0]
                for ant in range(3):
                    # plot the histogram of phase_diff of antenna in each packet
                    # plt.hist(phase_diff_modified_per_ant[ant], bins=40, density=True)
                    plt.hist(phase_diff_modified_per_ant[ant], bins=40)
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
                    plt.show()

                def phasetoangle(phase):
                    angle = float(phase/90)
                    angle = min(angle,1)
                    angle = max(angle,-1)
                    return math.asin(angle)/math.pi*180
                angles = [phasetoangle(k) for k in ant_result_diffs16]
                print("ant diffs[calculated by diff16-(avg_diff1)*16]", angles, 'avg', np.average(angles))
                diff3 = ((ant_result_diffs16[0]+ant_result_diffs16[1])/2-ant_result_diffs16[2])/3
                print('ant diffs[calculated by (updiff-downdiff)/3]:',phasetoangle(diff3))
                # plt.show()

                # plot phase_diff_modified
                # plt.plot(phase_diff_modified, 'r.', linewidth=0.5, markersize=3)
                # plt.show()
                # plt.hist(phase_diff_modified, bins=400, density=True, alpha=0.7)
                # plt.show()

                # find normal difflength
                # parameter: 24 for normal angles
                # samp = [j for j in phase_diff_modified if abs(j - 24) < 10]
                # diffconst = length * np.average(samp)
                # print(np.average(samp), len(samp), len(phases_all), "diffconst with length", diffconst)

                # draw lines for the original phases_plus
                for i in range(length, int(len(phases_plus)) - length, length):
                    # parameter: only use the mid length/2 points for linear regression
                    lenside = int(length / 4)

                    # do the linear_regression. the result line: y = a0 + a1 * j
                    a0, a1 = linear_regression(np.array(range(lenside, length - lenside)),
                                               np.array(phases_plus[i + lenside:i + length - lenside]))
                    plt.plot(np.array(range(i, i + length)), [a0 + a1 * (x - i) for x in range(i, i + length)],
                             'r.', linewidth=0.5, markersize=3)  # , label='phase calculated from I/Q output')
                    plt.plot(np.array(range(i, i + length)), phases_plus[i:i + length], 'b.', linewidth=0.5,
                             markersize=3)
                plt.title("phases and lines - antenna feature modified")
                plt.show()

                # draw lines for phases_plus with antenna diff removed
                # phase_diff_modified_sum = [sum(phase_diff_modified[:i + 1]) for i in range(len(phase_diff_modified) - 1)]
                for i in range(length, int(len(phases_plus_modified)) - length, length):
                    # parameter: only use the mid length/2 points for linear regression
                    lenside = int(length / 4)

                    # do the linear_regression. the result line: y = a0 + a1 * j
                    a0, a1 = linear_regression(np.array(range(lenside, length - lenside)),
                                               np.array(phases_plus_modified[i + lenside:i + length - lenside]))
                    plt.plot(np.array(range(i, i + length)), [a0 + a1 * (x - i) for x in range(i, i + length)],
                             'r.', linewidth=0.5, markersize=3)  # , label='phase calculated from I/Q output')
                    plt.plot(np.array(range(i, i + length)), phases_plus_modified[i:i + length], 'b.', linewidth=0.5,
                             markersize=3)
                plt.title("phases and lines - antenna feature modified")
                plt.show()

                phases_modified_diff_len = [phases_plus_modified[j + length] - phases_plus_modified[j] for j in
                                            range(0, len(phases_plus_modified) - length)]

                plt.plot(phases_modified_diff_len, 'b.', linewidth=0.5, markersize=3, label="modified phase")
                plt.plot([i - np.average(ant_result_diffs)*length for i in phases_diff_len], 'r.', linewidth=0.5, markersize=3, label="original phase")
                plt.title('phases_plus_modified diff across length phase[i+length] - phase[i]')
                plt.legend()
                plt.show()


