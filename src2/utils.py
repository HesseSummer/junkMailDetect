import re
import numpy as np
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import pickle
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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


"""格式转化"""
"""将y_val转为y_val_label"""
# y_val[0:3]示例：[[0. 1.]
#                 [1. 0.]
#                 [1. 0.]]
# y_val_label[0:3]示例：['spam' 'ham' 'ham']
def TOy_val_label(y_val):
    y_val_label = []
    spam = np.array([0., 1.]) ## [0 1]表示垃圾邮件？
    ham = np.array([1., 0.]) ## [1 0]表示正常邮件
    for line in y_val:
        if all(line == spam):
            y_val_label.append("spam")
        else:
            y_val_label.append("ham")
    
    y_val_label = np.array(y_val_label)
    return y_val_label
"""将y_pred转为y_pred_label"""
# y_pred[0:3]示例：[[9.9905199e-01 9.4802002e-04]
#                  [9.8692465e-01 1.3075325e-02]
#                  [1.0000000e+00 0.0000000e+00]]
# y_pred_label[0:3]示例：['ham' 'ham' 'ham']
def TOy_pred_label(y_pred):
    y_pred_label = []
    y_pred_index = np.argmax(y_pred, axis=1)

    for line in y_pred_index:
        if line == 0:
            y_pred_label.append("ham")
        else:
            y_pred_label.append("spam")
    
    y_pred_label = np.array(y_pred_label)
    return y_pred_label

# 模型部分
#
#



def saveModel(clf, name, config):
    base_path = config['base_path']
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    file_path = os.path.join(base_path, name+'.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(clf, f)
    print(name + "模型保存成功")


def getModel(name, config):
    base_path = config['base_path']
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



def showModel(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print(name + '的混淆矩阵：')
    print(cm)
    print(name + '的分类结果：')
    print(cr)


def print_AUC(y_val_label, y_pred):
    y_scores = y_pred[:, 1]
    fpr,tpr,threshold = roc_curve(y_val_label, y_scores, pos_label='spam')
    roc_auc = auc(fpr,tpr)
    
    plt.figure()
    lw = 2
    plt.figure(figsize=(5,5))
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

def saveTokenizer(tokenizer, config):
    file_path = config['tokenizer_path']
    with open(file_path, 'wb') as f:
        pickle.dump(tokenizer, f)

def loadTokenizer(config):
    file_path = config['tokenizer_path']
    with open(file_path, 'rb') as f:
        return pickle.load(f)
