from math import *


def iat2(y, x):
    # input: 0 < y / x < 1
    # output: arctan(y/x) in range(0, 64)
    # angle = 0 degrees: iat2(0 / 1) = 0
    # angle = 45 degrees: iat2(1 / 1) = 64

    # when using actual arctan:
    return atan(y / x) / (pi / 4) * 64
    # when using linear approximation:
    # return 64 * y / x


def iatan2sc(y, x):
    # return the angle of x, y representing the 360 degrees with range of [-128, 128]
    # from x+ axis clockwise: 0 -> 64 -> 128, -128 -> -64 -> 0

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


# return the angle difference between point(Xre, Xim) and (Yre, Yim)
# NOTE: (Xre, Xim) belongs to point 1 and (Yre, Yim) belongs to point 2
# DO NOT confuse with X, Y axis
def AOA_AngleComplexProductComp(Xre, Xim, Yre, Yim):
    Zre = Xre * Yre + Xim * Yim
    Zim = Xim * Yre - Xre * Yim
    # print(Zre,Zim)
    angle = iatan2sc(Zre, Zim)
    return angle
