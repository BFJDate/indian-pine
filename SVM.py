import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler  # 归一化
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import cross_val_score,cross_val_predict
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import copy
import re
# np.set_printoptions(threshold=np.inf)

#读取数据
def loadData():
    data=loadmat(os.path.join('.\\indian pine\\Indian_pines_corrected.mat'))['indian_pines_corrected']
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
train_data,test_data,train_labels,test_labels = train_test_split(data_stand ,labels ,test_size=0.2 ,random_state=24,stratify=labels)

# #主成分分析特征提取
# k = 15
# pca = PCA(n_components=k , svd_solver='randomized' ,whiten=True).fit(data_stand)
# train_data_pca =pca.transform(train_data)
# test_data_pca =pca.transform(test_data)
# data_pca =pca.transform(data_stand)


# #fisher线性分析降维
# l = 15
# lda = LinearDiscriminantAnalysis(n_components=l).fit(train_data_pca,train_labels)
# train_data_pca_lda = lda.transform(train_data_pca)
# test_data_pca_lda = lda.transform(test_data_pca)
# data_pca_lda = lda.transform(data_pca)

def Classfiction(train_data,test_data,train_labels,test_labels,data,method):
    # i = 1
    # best_score = 0
    # # 网格搜索
    # for gamma in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
    #     for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    #         clf = svm.SVC(kernel = 'rbf' ,gamma=gamma, C=C)  # 对于每种参数可能的组合，进行一次训练；
    #         clf.fit(train_data, train_labels)
    #         score = np.mean(cross_val_score(clf,train_data,train_labels,cv=5))
    #         Predict = cross_val_predict(clf,data_stand,labels,cv=5)
    #         if score > best_score:  # 找到表现最好的参数
    #             best_score = score
    #             best_parameters = {'gamma': gamma, 'C': C}
    #         print("实验{}的第{}次分类，当前正确率为：sceore = {},参数为：gamma = {}，C = {};最高正确率为：sceore = {},最优参数为：gamma = {}，C = {}".format(
    #             method,i,score,gamma,C,best_score,best_parameters['gamma'],best_parameters['C']))
    #         i = i + 1
    #         # return Predict

    # 结果优化
    a = 1    #池化范围大小
    clf = svm.SVC(gamma=1, C=10)  # 对于每种参数可能的组合，进行一次训练；
    clf.fit(train_data, train_labels)
    print(cross_val_score(clf,test_data,test_labels,cv=5))
    Predict = cross_val_predict(clf,data_stand,labels,cv=5)
    # # score = clf.score(test_data, test_labels)
    Predict_2 = Predict.reshape(145, 145)
    np.save("Predict_2.npy", Predict_2)

    for i in range(a-1, 146 - a):
        for j in range(a-1, 146 - a):
            count = np.zeros(17)
            arr = []
            for m in range(i - a,i + a+1):
                for n in range(j - a,j + a+1):
                    try:
                        arr.append(Predict_2[m][n])
                    except:
                        pass
                    # print(Predict_2[i][j])
                    # print(arr)
            for k in range(len(arr)):
                for target in range(0, 17):
                    if arr[k]==target:
                        count[target] = count[target]+1


            b = count.argmax()
            Predict_2[i][j] = b


    Predict_1 = Predict_2.reshape(-1)
    train_data_1, test_data_1, train_labels_1, test_labels_1 = train_test_split(data_stand, Predict_1, test_size=0.2,
                                                                                random_state=24)
    # 查看错误率
    ErrorCounter = 0
    for i in range(len(test_labels)):
        if (test_labels_1[i]) != test_labels[i]:
            ErrorCounter = ErrorCounter + 1
    print(ErrorCounter)
    print(1 - ErrorCounter / len(test_labels))

    return Predict_2



test_time = 1
clf = Classfiction(train_data,test_data,train_labels,test_labels,data_stand,1)
# clf = Classfiction(train_data_pca,test_data_pca,train_labels,test_labels,data_pca,2)
# clf = Classfiction(train_data_pca_lda,test_data_pca_lda,train_labels,test_labels,data_pca_lda,3)

# clf = clf.reshape(145,145)
labels = labels.reshape(145,145)
print(type(labels))
plt.figure(figsize=(8, 6))
plt.imshow(clf, cmap='viridis')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Predictive image')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(labels, cmap='viridis')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('original image ')
plt.show()

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
    
draw_confusion_matrix(label_true=clf,  # y_gt=[0,5,1,6,3,...]
                      label_pred=labels,  # y_pred=[0,5,1,6,3,...]
                      label_name=["0", "1", "2", "3", "4", "5", "6"
                                  , "7", "8", "9", "10", "11"
                                  , "12", "13", "14", "15", "16"],
                      title="Confusion Matrix",
                      pdf_save_path="Confusion_Matrix.jpg",
                      dpi=300)   