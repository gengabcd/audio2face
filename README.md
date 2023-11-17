# audio2face

## Environment

pytorch-cpu >= 2.0

scipy

pandas

## audio2face

该模型复现了nvidia的论文。测试代码在audio2face/audio2face/modeltest.py中。根据注释修改代码即可。前端演示为unity

模型文件res.zip中包含了许多次测试。res文件的结构如下：

res--

​	--train0

​	----dir: a lr = 0.0001 小样本训练 单一loss: 训练描述

​	----checkpoint：训练模型权重

​	------model_epochs_x.pth: 训练模型权重文件，epochs_x便是训练了多少x*10 epochs

​	----res.json:训练集loss和测试集loss的json文件，数组下标表述epoch轮数

3D-ETF.zip是数据集，包含了许多音频文件和对应的参数

文件链接：

链接：https://pan.baidu.com/s/1iNCJ-dt_8PXa8eaGBWsY1g 
提取码：virt 
--来自百度网盘超级会员V4的分享



​	




