import os
from scipy.io import loadmat
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor,export_graphviz
import numpy as np
import graphviz  
import matplotlib.pyplot as plt
os.environ['PATH'] = os.pathsep + r'E:\deep learning ai\ana\Graphviz\bin'

#读取数据
def loadData():
    data=loadmat(os.path.join('indian pine\\Indian_pines_corrected.mat'))['indian_pines_corrected']
    labels=loadmat(os.path.join('indian pine\\Indian_pines_gt.mat'))['indian_pines_gt']
    return data ,labels
#数据标准化
data,labels = loadData()
data_2d = data.reshape(data.shape[0]*data.shape[1],data.shape[2])
labels = labels.reshape(-1)
data_stand = np.zeros(data_2d.shape)
data_stand= MinMaxScaler().fit_transform(data_2d)

data1 = data_stand.reshape(145,145,200).tolist()


a = 1
for a in range(1,5):
    # print(np.shape(data))
    for i in range(0, 145):
        for j in range(0, 145):
            mean = 0
            for m in range(i - a, i + a + 1):
                for n in range(j - a, j + a + 1):
                    try:
                        if labels[m][n]!=0:
                            sum = np.sum(data[m][n])
                            mean = mean + sum/len(data[m][n])
                    except:
                        pass

            # print(np.shape(data))
            data1[i][j].append(mean)
            # print(np.shape(data))

data_stand = np.array(data1).reshape(145*145,204)
#划分数据集
train_data,test_data,train_labels,test_labels = train_test_split(data_stand ,labels ,test_size=0.2 ,random_state=24)

#主成分分析特征提取
k = 15
pca = PCA(n_components=k , svd_solver='randomized' ,whiten=True).fit(data_stand)
train_data_pca =pca.transform(train_data)
test_data_pca =pca.transform(test_data)
data_pca =pca.transform(data_stand)
#fisher线性分析降维
l = 15
lda = LinearDiscriminantAnalysis(n_components=l).fit(train_data_pca,train_labels)
train_data_pca_lda = lda.transform(train_data_pca)
test_data_pca_lda = lda.transform(test_data_pca)
data_pca_lda = lda.transform(data_pca)
#CART决策树
clf = DecisionTreeClassifier(criterion="gini" #采用CART
                                  ,random_state=30
                                  ,splitter="best" 
                                  ,max_depth=5 #最大深度
                                  ,min_samples_leaf=5 #一个节点分支后，每一个子节点至少包含5个样本
                                  ,min_samples_split=5 #一个节点至少包含5个样本才会分支
)
clf1 = DecisionTreeClassifier()
clf.fit(train_data,train_labels)
print(clf.score(train_data,train_labels))   # 训练集准确率
print(clf.score(test_data, test_labels))   # 测试集准确率
final = clf.predict(data_stand)
#绘制决策树
feature_names=[]
class_name=['background','Alfalfa','Corn-notill','Corn-mintill','Corn','Grass-pasture','Grass-trees','Grass-pasture-mowed','Hay-windrowed','Oats','Soybean-notill','Soybean-mintill','Soybean-clean','Wheat','Woods','Buildings-Grass-Trees-Drives','Stone-Steel-Towers']
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=feature_names,
                     class_names=class_name,filled=True,rotate=True,rounded=True,
                     special_characters=True)
#graph = graphviz.Source(dot_data)
#graph.view()

#绘制标签图
final = final.reshape(145,145)
labels = labels.reshape(145,145)
plt.figure(figsize=(8, 6))
plt.imshow(final, cmap='viridis')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('CART Predictive image')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(labels, cmap='viridis')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('original image ')
plt.show()

from sklearn.metrics import confusion_matrix
def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            if cm[j][i] >= 0.5:
                color = (1, 1, 1)
            else:
                color = (0, 0, 0)
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
    
draw_confusion_matrix(label_true=final,  # y_gt=[0,5,1,6,3,...]
                      label_pred=labels,  # y_pred=[0,5,1,6,3,...]
                      label_name=["0", "1", "2", "3", "4", "5", "6"
                                  , "7", "8", "9", "10", "11"
                                  , "12", "13", "14", "15", "16"],
                      title="Confusion Matrix",
                      pdf_save_path="Confusion_Matrix.jpg",
                      dpi=300)   