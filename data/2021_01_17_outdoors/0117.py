# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from math import *
import matplotlib.pyplot as plt


def iat2(y, x):
    assert (y / x >= 0)
    assert (atan(y / x) / (pi / 4) <= 1)
    return (atan(y / x) / (pi / 4) * 64 )
    # print(atan(y/x)*180/3.14)
    # return length * (y / x)  # / (pi / 4)


def iatan2sc(y, x):
    if (y >= 0):  ## oct 0,1,2,3
        if (x >= 0):  ## oct 0,1
            if (x > y):
                return iat2(-y, -x) / 2 + 0 * 32
            else:
                if (y == 0): return 0  # (x=0,y=0)
                return -iat2(-x, -y) / 2 + 2 * 32

        else:  # oct 2,3
            # if (-x <= y) :
            if (x >= -y):
                return iat2(x, -y) / 2 + 2 * 32
            else:
                return -iat2(-y, x) / 2 + 4 * 32


    else:  # oct 4,5,6,7
        if (x < 0):  # oct 4,5
            # if (-x > -y) :
            if (x < y):
                return iat2(y, x) / 2 + -4 * 32
            else:
                return -iat2(x, y) / 2 + -2 * 32

        else:  # oct 6,7
            # if (x <= -y) :
            if (-x >= y):
                return iat2(-x, y) / 2 + -2 * 32
            else:
                return -iat2(y, -x) / 2 + -0 * 32


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, :name')  # Press Ctrl+F8 to toggle the breakpoint.


def AOA_AngleComplexProductComp(Xre, Xim, Yre, Yim):
    Zre = Xre * Yre + Xim * Yim
    Zim = Xim * Yre - Xre * Yim
    # print(Zre,Zim)
    angle = iatan2sc(Zre, Zim)
    return angle


import numpy as np


def linear_regression(x, y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x ** 2)
    sumxy = sum(x * y)
    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
    return np.linalg.solve(A, b)
import os
length = 16
lenside = int(length/4)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for i in range(0, 360, 30):
        a = i * pi / 180 + 0.00001
        print(a, cos(a), sin(a), pi / 2 - atan(cos(a) / sin(a)), pi / 2 - iatan2sc(cos(a), sin(a)) * (pi / 4) / 32)
    filepath = r"D:\prog\ti\simplelink_cc13x2_26x2_sdk_4_30_00_54\tools\ble5stack\rtls_agent\examples\rtls_aoa_iq_with_rtls_util_export_into_csv_log"
    filepath = r"D:\desktop\rtls_agent\examples\rtls_aoa_iq_with_rtls_util_export_into_csv_log"
    filepath = r"D:\prog10\Desktop\aoadata\rtls_agent\examples\rtls_aoa_iq_with_rtls_util_export_into_csv_log"
    filename = os.listdir(filepath)[-1]
    print(filename)
    with open(os.path.join(filepath,filename)) as f:
        with open('D:\\prog10\\desktop\\1o.csv','w') as g:
            xs = []
            ys = []
            for line in f.readlines()[1:]:
                # print(i)
                identifier,pkt,sample_idx,rssi,ant_array,channel,i,q = line.split(',')
                xs.append(int(i))
                ys.append(int(q))
            results = [iatan2sc(ys[i], xs[i]) * 180 / 128 for i in range(len(xs))]
            results = results[32:]

            # results =[AOA_AngleComplexProductComp(xs[i+1],ys[i+1],xs[i],ys[i]) for i in range(len(xs)-1)]
            res2 = []
            i = 0
            diff = 0
            for j in range(len(results) - 1):
                res2.append(results[i + j] + diff)

                temp = results[i + 1 + j] - results[i + j]
                if abs(temp) > 180:
                    diff += -temp / abs(temp) * 360
            samp = []
            res2diff = [res2[j+1]-res2[j] for j in range(0,len(results)-2)]
            for di in range(-360,360,40):
                print(di,len([k for k in res2diff if abs(k-di)<20]))
            res2new = []
            for j in range(1,len(res2)):
                if abs(res2[j]-res2[j-1])>20:res2new.append(res2[j])
            # results = res2new
            # res2 = res2new
            _X2 = np.array(range(len(res2diff)))
            plt.plot(_X2, res2diff, 'r', linewidth=0.5, markersize=1)
            plt.show()
            plt.hist(res2diff, bins=400, density=True, alpha=0.7)
            plt.show()
            for j in res2diff:
                if(abs(j-75)<20):samp.append(j)
            print(samp)
            print(np.average(samp),len(samp),len(results))

            ress=[]
            for i in range(length, len(results)-length, length):

                diff = 0

                a0, a1 = linear_regression(np.array(range(lenside, length-lenside)), np.array(res2[i + lenside:i + length - lenside]))
                temp3 = -1
                ot1 = 0
                ot2 = 0
                for j in range(length):
                    temp = res2[i + j] - (a0 + a1 * j)
                    temp2 = temp
                    if j != 0: temp2 = 0
                    temp4 = res2[i + j]-res2[i + j - 1]
                    if( abs(temp4-temp3)>180):

                        temp5 = temp4
                        temp4 += 360 * -(temp4-temp3) / abs(temp4-temp3)
                        temp3 = temp5
                        # if(temp4<-270):temp4+=360
                    #if (temp4 < -90): temp4 += 360
                    if(temp4>0):ot1 = temp4
                    else:ot2=temp4
                    g.write(f'{xs[i + j]},{ys[i + j]},{temp},{temp2},{results[i+j]},{res2[i+j]},{a0 + a1 * j},{temp4},{ot1},{ot2}\n')

                _X2 = np.array(range(i, i + length))
                _Y2 = [a0 + a1 * (x - i) for x in _X2]
                # 显示图像
                res3 = [res2[k] - res2[k-length] for k in range(i,i + length)]
                plt.plot(_X2, res3, 'r.', linewidth=0.5,markersize=1)  # , label='phase calculated from I/Q output')
                ress.extend(res3)
                # plt.plot(_X2, res2[i:i + length], 'g.', markersize=1)  # , label='phase calculated from I/Q output')
                # plt.plot(_X2, _Y2, 'b.', markersize=1)  # , label='linear regression', color='C1')
            plt.xlabel('angle/degrees')
            plt.ylabel('number of samples')
            plt.legend()
            plt.show()

            plt.hist(ress,bins=400,density=True,alpha=0.7)
            plt.show()

            # break
            # s = AOA_AngleComplexProductComp(xs[i+1],ys[i+1],xs[i],ys[i])
            # s = s * (pi / 4) / 32
            # s = s * 180 / pi
            # print(90-s)

            # print(results)
            # import matplotlib.pyplot as plt

            # x = range(len(results))
            # y = results
            # 有了 x 和 y 数据之后，我们通过 plt.plot(x, y) 来画出图形，并通过 plt.show() 来显示。
            # plt.plot(x, y)
            # plt.show()
    # See PyCharm help at https:#www.jetbrains.com/help/pycharm/

