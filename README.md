# 2017知乎看山杯机器学习挑战赛 

## Koala队伍解决方案

### 运行环境

基于Python2及PyTorch，需安装：
1. PyTorch
2. Numpy
3. visdom
4. fire

运行：
```shell
cd src; pip2 install -r requirements.txt
python2 -m visdom.server
```

### 数据分析

代码在data_analysis文件夹下，分析结果见链接：

### 数据处理

代码在data_preprocess文件夹下，question_preprocess、label_preprocess、topic_preprocess，分别有对应的notebook和py版本。

处理后的全部数据文件链接：

解压后放置在data_preprocess中

### 单模型训练

代码在src文件夹下，需在其中新建snapshots文件夹，用于存储模型文件

- dataset：存放数据load文件
- models：存放所有模型定义文件，主要用到FastText.py、TextCNN.py、RNN.py
- utils：存放工具文件，如模型加载与保存、日志、可视化、矩阵处理等
- config.py：配置文件，可在运行时通过命令行修改
- main.py：所有程序入口

在src下运行：
```shell
python2 main.py train --model=RNN --use_word=True --batch_size=256
```

上述命令后面都是可设置的参数
- model是使用的模型（与models下文件名一致，结果保存在snapshots/模型名/）
- use_word表示使用word训练，如使用char，则改为--use_char=True
- batch_size表示训练batch，显存不够的可以适当减小
- 还有其他一些参数，见config.py中的配置

### Boosting模型训练

对于单个模型来说，其所能实现的效果毕竟有限。通过分析数据，我们发现一个模型对于不同类别是具有偏向性的，即有的类可能会全部预测错，而另一个类则会全部预测对，这种类别之间的差异性对预测性能会有很大的影响
因此，我们针对这种偏差，借鉴Boost提升的思想，提出了一个新颖的做法，对结果进行修复性训练多层并累加。

具体的，对于一层训练的结果，计算出其类别的错误率，计算公式如下：

loss_weight[i] = 第i个类被标记了但没有出现在预测的top5中的样本数目/有多少样本标记了第i个类

而后根据此错误率作为代价敏感，继续训练下一层，得到下一层结果后，与上一层累加，再计算错误率，以此往复，后面的层都是在修复前面层的偏差，最后的预测是使用所有层的累加结果。

具体计算代码在main.py的get_loss_weight函数中

在src下运行：
```shell
python2 main.py train --model=RNN --use_word=True --batch_size=256　--boost=True --base_layer=0
```

将base_layer依次改为1、２、３...，可逐层训练，训练的累加结果保存在与模型同目录，文件名包含cal_res，当前层的loss_weight也存于此处

### 测试

- 加载训好的模型并测试：参考gen_test_res.py
- 直接融合各模型测试的结果文件：参考utils/resmat.py
