# 预处理部分
#
#
import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
"""
从文本到文本，使用正则去除标点、缩写、html符号
"""
def regUse(text):
    text = re.sub(r"[,.?!\":]", '', text) # 去标点
    text = re.sub(r"'\w*\s", ' ', text) # 去缩写
    text = re.sub(r"#?&.{1,3};", '', text) # 去html符号
    return text.lower()


"""
简单分词：从文本到单词列表，去除停用词
"""
def sampleSeg(text):
    tokens = [word for word in word_tokenize(text) if word not in stopwords.words('english') and len(word)>=3]
    return tokens


"""
辅助函数，辅助lemSeg：从字符串到wordnet的词性
"""
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


"""
词形分词：调用pos_tag为tokens中的单词标注词性，被lemmatize利用，进一步分词
"""
def lemSeg(tokens):
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(tokens):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return res


"""
将上述函数组合在一起的预处理
"""
def preprocess(text):
    text = regUse(text)
    tokens = sampleSeg(text)
    tokens = lemSeg(tokens)
    return tokens  ## 返回的是单词列表


# 模型部分
#
#

import pickle
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def saveModel(clf, name, config):
    base_path = config['base_path']
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    file_path = os.path.join(base_path, name+'.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(clf, f)
    print(name + "模型保存成功")


def getModel(name, config):
    base_path = config['base_config']
    file_path = os.path.join(base_path, name+'.pkl')
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except EOFError:  # 捕获异常EOFError 后返回None
        print('错误：尝试读取空文件')
        return None


# 画图部分
#
#


import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc

def showModel(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print(name + '的混淆矩阵：')
    print(cm)
    print(name + '的分类结果：')
    print(cr)


def printAUC(clf, x_test, y_test):
    y_probas = cross_val_predict(clf, x_test, y_test, cv=3, method="predict_proba")
    print(y_probas[0:5])
    y_scores = y_probas[:, 1]
    print(y_scores[0:5])
    print(y_test[0:5])
    fpr, tpr, threshold = roc_curve(y_test, y_scores, pos_label='spam')
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def printAUCsvm(clf, x_test, y_test):
    y_score = clf.decision_function(x_test)

    fpr, tpr, threshold = roc_curve(y_test, y_score, pos_label='spam')
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# 其他
#
#

def logging(title, content):
    print('-------- {} --------'.format(title))
    print(content)
