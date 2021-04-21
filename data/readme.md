### 数据采集

* 2021_01_17 在室内各处收集数据，当时把板子放在地上或桌上，因此可能收到了大量multipath的干扰。

  ​	ground truth: 粗略目测45°。

  ​	实际结果：

  ​	第一个包：虽然不好看，但是平均出来是47°，异常地准确。

  ![image-20210315100619261](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210315100619261.png)

  后续的包有的形状不太一样，比如这是第三个包，结果飘到了56°。

  ![image-20210315100114226](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210315100114226.png)

  ​	注意，如果算出来是45°，这`phase[i+16]-phase[i]`应该为$90/\sqrt{2}*3$，在190°左右。

  ​	所有包的数据如下。均值49°，不过由于本身ground_truth也不一定是45°，不太知道其真正的准确性。

  ![image-20210315101616663](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210315101616663.png)

  ​	不过之前做的antenna频率不同的校正并没有什么效果。

  ![image-20210315102016761](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210315102016761.png)

* 2021_01_18 晚上6点 在室外美院前收集数据。依然没有使用三脚架，是放在石墩子上做的。ground truth: 不明。当时主要在解决断开连接的问题，随意摆放了片子，并没有朝向slave，并且采样时候也没有持稳板子，没什么后续研究意义。

  ![image-20210315103049143](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210315103049143.png)

* 2021_01_19 下午 继续在室外收集。依然没有使用三脚架。ground truth: 粗略目测45°。

* ![image-20210315103207695](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210315103207695.png)

  这时候出现了非常严重的倾斜，导致之后投入精力研究这一问题，但看起来这可能是个例。

* 2021_01_11 跟良哥汇报（ppt亦已上传）。

* 2021_03_04 在室外用铁架子搭着做了。但是由于之前在尝试用master做改变切换天线周期的尝试，**当时负载天线阵列的板子其实烧录了master固件**，虽然MAC地址正确，但是数据全都来自Master。ground truth为45°，由激光测距仪获得。距离较远，大概在6米以上。

* 2021_03_12 在室内走廊上收集了两组数据。一组是45°，一组是35°，用一根网线和量角器测量角度，都有准确的 ground truth。距离为380mm。_2021_03_12_45degrees_indoors_文件夹存放了调效后的最后五组数据，_archives_中的_2021_03_12_45degrees_indoors_buggy_文件夹是那之前采集的所有数据，有很多bug了的空文件，供参考。

* 2021_03_16 沙尘暴之后的第二天 在楼外的小花园过道里采了组数据。45°是目测的，距离6米。注意，换了个新的板子（passive标签斜贴在三角板上的那个），所以mac地址不同。每一个数据包都有很大的区别，从30°到45°到55°疯狂波动。

* 2021_03_27 小雨，下午4点雨停了之后去小花园收集数据。0~90°每个angle一组。把新的passive摔坏了，忏悔！改用旧的。空气湿度极高，不知道会不会有影响。

  倒叙读一遍。

  2度 packet0 [i+1]-[i]

  想到 这里每48左右一个大峰应该是噪音吧 可以去掉 不知道为什么会出现 可能是2~0切换时间长了 卡了一下子

  ![image-20210327024840840](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327024840840.png)

![image-20210327025636935](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327025636935.png)

峰全在ant0上，那大概是2~0卡了。ant2最后一个点收的好好的，切换时候卡了，0的第一个点就gg了。

2和3都非常正常

![image-20210327030057531](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327030057531.png)

**奇怪的点是 峰并不在整数点上（红标）**

![image-20210327030114127](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327030114127.png)

包成这个样子基本上是废掉了。不过滤掉

![image-20210327030127543](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327030127543.png)

不太好修正呐...

![image-20210327033738658](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327033738658.png)

![image-20210327033946907](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327033946907.png)

无论是这样子把每一个偏离点（红色）变成上一个点，

![image-20210327034242793](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327034242793.png)

![image-20210327034406783](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327034406783.png)

效果见上2图，

![image-20210327034126969](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327034126969.png)

还是这样子变成22，

![image-20210327034146857](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327034146857.png)

![image-20210327034434486](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327034434486.png)

效果都不好。**对修正值的影响，只会平移上图这个红色带，不能把他抹平。**不太清楚为什么。

![image-20210327035237079](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327035237079.png)

第二个packet尾巴坏掉了 截掉尾巴之后算法一个是10°一个是-6°。

![image-20210327035303449](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327035303449.png)

第三个 4°和16°

第四个 又开始有离群点了

![image-20210327035353092](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327035353092.png)

![image-20210327035404531](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327035404531.png)

数据倒是很好看

算出来10°和-29°

下面是所有包的结果（前半部分 后面乱七八糟的了）

![image-20210327035955947](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327035955947.png)

蓝色是分别算angle平均起来 似乎好看点儿 但是这groundtruth在2°可能本身就不容易测准吧。

15°truth的

![image-20210327040251355](C:\Users\lm\AppData\Roaming\Typora\typora-user-images\image-20210327040251355.png)