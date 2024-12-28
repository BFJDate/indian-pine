# indian-pine
使用各种方法对indian-pine数据集进行训练分类

Indian Pines 是最早的用于高光谱图像分类的测试数据，由机载可视红外成像光谱仪（AVIRIS）于 1992 年对美国印第安纳州一块印度松树进行成像，然后截取尺寸为 145×145 的大小进行标注作为高光谱图像分类测试用途。

Indian pine原标签图
![image](https://github.com/user-attachments/assets/2d7d7b4b-cb07-4baa-8e24-0f3c35572fbe)

# —.CART决策树
使用CART决策树进行分类，剪枝规则最大深度为5一个节点分支后，每一个子节点至少包含5个样本，且一个节点至少包含5个样本才会分支，可以得到较好的效果。
数据经过主成分分析进行降维，使用80%的数据进行训练。训练集和测试集准确率分别为0.64179548156956和0.6375743162901308。
决策树图像

![image](https://github.com/user-attachments/assets/e89f575f-ebd0-42a4-a117-8ebc3657195c)

经CART分类后的标签图

![image](https://github.com/user-attachments/assets/81272968-eb2d-42a8-b191-af12604c13d0)

# 二.SVM分类器
SVM分类器采用OvR（One-vs-the-Rest）分类策略,采用网格搜索（Grid Search）的方法寻找参数gamma与c的值。
使用80%数据作为训练集，当gamma=1，c=10时，clf1准确率最高，为：0.8420927467300833。此时clf2的正确率为：0.6059453032104637，clf3的准确率为：0.6007134363852557。

当gamma=0.01，c=时100，clf2与clf3的准确率最高，clf2为0.7545778834720571, clf3为：0.7621878715814506。此时clf1的正确率为：0.7807372175980976
经SVM分类后的标签图

![image](https://github.com/user-attachments/assets/2127c9ea-942a-4b17-9539-47d7f69337a9)

# 三 神经网络方法
使用VGG11网络构建Indian Pines数据识别分类器，完成网络搭建后开始启动训练。
使用20%的数据作为训练集，训练在GPU上进行，运行速度约为1s/epoch，训练了150个epoch后，验证集的准确率约为78.34%，Train_loss降到了0.0008，Test_accuracy约为84.83%
训练结果

![image](https://github.com/user-attachments/assets/3d44cae0-818a-4aa4-9ae7-a2ce835b927f)

# 改进
在神经网络方法的基础上，对数据添加位置特征，每个点加了3个特征，第一个是该点周围3x3范围内的特征平均值；第二个是该点周围5x5范围内的特征平均值；第三个是该点周围7x7范围内的特征平均值。同时使用5折交叉验证的方法进行训练

# 最终结果得到最高平均正确率为0.96823

混淆矩阵

![image](https://github.com/user-attachments/assets/26641010-90af-4252-8b0d-3111135ed91d)

预测结果图

![image](https://github.com/user-attachments/assets/e15e71c5-8acf-480c-847d-e7acf65e9a97)

