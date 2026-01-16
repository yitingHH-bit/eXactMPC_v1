MPC模块使用方法：

1.simulink选取MPC Controller

![](https://i0.hdslb.com/bfs/note/db5b8984586f4f38ed73c13e9495cb9dd79e4ef8.jpg@690w_!web-note.webp)

mo:测量输出量

ref:参考输入量

md:测量扰动量

mv:操纵变量

  

2.模块内调节各项参数

![](https://i0.hdslb.com/bfs/note/f323bc5244b07b5538f35a51ff5442287ee651fe.jpg@690w_!web-note.webp)

上半灰底：参数部分，可以进行mpc控制器的参数设计以及初始状态的设置

下半白底：模块选项，依次为通常选项，在线特性，默认条件，其他

通常选项：

外加输入：可测量扰动，外部操纵变量（可增减）

外加输出：可添加各项输出特性参数（暂不考虑）

![](https://i0.hdslb.com/bfs/note/dcd3f0e9349288df551f641e01b64332738db0f2.jpg@690w_!web-note.webp)

  

状态估计：是否自定义估计状态

在线特性：可设置常量，权重，可操纵变量的目标值

![](https://i0.hdslb.com/bfs/note/3f9325699e12b57474c3927fa3096d79db5b63f4.jpg@690w_!web-note.webp)

默认条件：设置采样时间，各种输入输出，参量个数

![](https://i0.hdslb.com/bfs/note/4d959e3e1d7d99d8e550204b5ab33baefb2a8027.jpg@690w_!web-note.webp)

其他部分：

可设置信号属性，数据模块的数据类型，采样时间以及最佳设定

![](https://i0.hdslb.com/bfs/note/d1019e70e24fbd7a490d516a2c4229e6ec0f7c84.jpg@690w_!web-note.webp)

3.实操

（1）定义mpc的结构，双击模块进入后点击design

![](https://i0.hdslb.com/bfs/note/098def994a21dd35fe3ad54e12782fc8eb1fa288.jpg@690w_!web-note.webp)

选择mpc structure

![](https://i0.hdslb.com/bfs/note/bbea984f3a7448e67c095a20d8ee658cfbb1eff5.jpg@690w_!web-note.webp)

选择change I/O size

![](https://i0.hdslb.com/bfs/note/941eb78a867d21af64c07bae6ac83814c94cd7ae.jpg@690w_!web-note.webp)

设置采样时间

![](https://i0.hdslb.com/bfs/note/a9d9b0ef878822f9f49f74c1c728aaf2fd2ffecf.jpg@690w_!web-note.webp)

下面设置输入输出仿真信号，一般已经设定好

![](https://i0.hdslb.com/bfs/note/811295dd102f97cac12130a15dd56e9efa04b48d.jpg@690w_!web-note.webp)

点击define and line

mpc1是系统自动创建的默认mpc控制器

scene1是系统默认的场景

![](https://i0.hdslb.com/bfs/note/b55b84ba25ca635063f901378c37a01f6384d576.jpg@690w_!web-note.webp)

（2）定义输入输出通道的属性

![](https://i0.hdslb.com/bfs/note/e2f295a3eaef08ffac4b81a210324a0f80ebf226.jpg@690w_!web-note.webp)

赋予输入输出变量有意义的名称和单位，标称值和比例因子保持默认即可，然后点击ok

![](https://i0.hdslb.com/bfs/note/28b91f1efbf547e65a22bb72464d545beb869cde.jpg@690w_!web-note.webp)

双击scene1，调整采样时间为100，下方调整步长和时间，保持默认即可，点击OK

![](https://i0.hdslb.com/bfs/note/d7899606f8d6b9ed8212407621f8ee4d85df0903.jpg@690w_!web-note.webp)

配置控制器的范围，设定输入输出上下限，点击constraints

![](https://i0.hdslb.com/bfs/note/729715047e215b77ab69a59a7367b36e8a3705f4.jpg@690w_!web-note.webp)

设定权重

![](https://i0.hdslb.com/bfs/note/140d92d1668bece4bb05b0b1337d054dd56f9413.jpg@690w_!web-note.webp)

调整超调量，左调降低超调

![](https://i0.hdslb.com/bfs/note/87f24cdf5e55ab2362dac98220b6fd8d52ae7b83.jpg@690w_!web-note.webp)

如下图保存

![](https://i0.hdslb.com/bfs/note/e17ae9308b3c5b8278f10a77e10dd4279306e0b2.jpg@690w_!web-note.webp)
