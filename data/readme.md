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