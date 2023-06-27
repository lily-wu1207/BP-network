# BP-network
handwritten BP-network using python3
## 说明
本实验中图片的格式采用的是.bmp，测试文件组织结构为：  
-1  
  -1.bmp  
  -2.bmp  
  -3.bmp  
  -...  
-2  
  -1.bmp  
  -2.bmp  
  -3.bmp  
  -...  
-...  
-12  
  -1.bmp  
  -2.bmp  
  -3.bmp  
  -...  
共12类，每类有若干张图片，在测试时将其转换成.npy格式文件，整理成<key,value>的格式并打乱顺序  
文件中model_91.26.npy为训练得到的模型参数

## 代码基本架构
### 数据预处理： split_data.py
将数据按9：1的比例划分为训练集和测试集。  
将原始的3通道的图象转化为单通道的灰度图，并展平为一维矩阵，利用np.save()函数存储为.npy格式文件：其中test_data.npy, train_data.npy文件分别存储测试集与训练集的图象数据，而test_label.npy, train_label.npy存储对应的标签值。  
### 手写BP网络：BP.py
#### 网络自定义函数：build_net
本实验的网络设计为可以灵活设置层数、神经元个数、学习率的网络。实现：  
a）自定义网络结构（层数，每层的神经元个数）：  
b）自定义损失函数：可选L2，CE（交叉熵）  
c）自定义每一层的激活函数：可选relu, sigmoid, no-active（无）  
build_net通过传入的输入参数维度，隐藏层的各层神经元个数来构建相应维度的权值矩阵，通过传入的激活函数类型确定每一层的激活函数，各层的偏置矩阵设为全为0.1的矩阵。  
#### Mini-batch训练：update_wb
可以通过batchsize调整batch的大小。每个batch计算完后更新权重：  
a）前向传播 feedforward: 对上一层的输入进行线性变换以及激活，并将每一层的输出记录在input_acfun（记录每一层线性层后的结果）以及input_layer（记录每一层的输入）中，便于后续逆向传播的计算。  
b）计算损失函数 loss_fun: 根据传入的损失函数计算对应的值  
c）反向传播 ：从后向前将损失delta反向传递，并将每一层的delta记录在数组deltas[]中，便于后续的权重更新。  
d）利用误差，修正每一层的权重：根据反向传播的误差，推出每一层的dw和db，根据学习率alpha对每一层的权重以及偏置进行调整。  
#### 在训练集和测试集上验证正确率：test_accuracy
调用feadforward模块，得到预测的概率分布lab_det，与真实的概率分布（one-hot编码）labs_true进行比较，得到出错的个数N_error，除以总数得到错误率error_rate。  

![image](https://github.com/lily-wu1207/BP-network/assets/105954052/b8f05574-ac92-42e2-9c43-ad92a80d09a5)

