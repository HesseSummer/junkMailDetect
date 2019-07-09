import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocess, logging, loadCleanTexts, saveCleanTexts, saveTokenizer, loadTokenizer
from config import config

def readData(config):
    file_path = config['raw_data']
    encoding = config['raw_encoding']
    with open(file_path, 'r', encoding=encoding) as f:
        raw_data = csv.reader(f, delimiter='\t')
        raw_data = list(raw_data)

    logging('原始数据示例', raw_data[0:3])
    return raw_data


def cleanData(raw_data):
    sms_text = []
    sms_label = []
    for line in raw_data:
        sms_text.append(" ".join(preprocess(line[1])))
        sms_label.append(line[0])
    logging('预处理后的文本示例', sms_text[0:3])
    logging('预处理后的标签示例', sms_label[0:3])
    return sms_text, sms_label

def vectorize(sms_text, sms_label, config):
    div_ratio = config['div_ratio']
    size_dataset = len(sms_text)
    size_trainset = int(round(size_dataset*div_ratio))
    logging('数据集的规模', size_dataset)
    logging('训练集的规模', size_trainset)
    logging('测试集的规模', (size_dataset-size_trainset))

    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), stop_words='english', strip_accents='unicode', norm='l2').fit(sms_text)
    ## saveCleanTexts(sms_text, config)
    saveTokenizer(vectorizer, config)

    x_train = np.array(sms_text[0:size_trainset])
    logging('文本形式的数据集示例', x_train[0:1])
    x_train = vectorizer.transform(x_train)
    logging('向量形式之稀疏矩阵示例', x_train[0:1])
    logging('向量形式之稠密矩阵示例', x_train[0:1].todense())
    y_train = np.array(sms_label[0:size_trainset])

    x_test = np.array(sms_text[size_trainset + 1: size_dataset])
    x_test = vectorizer.transform(x_test)
    y_test = np.array(sms_label[size_trainset + 1: size_dataset])

    return x_train, y_train, x_test, y_test


def getInputvec(raw_sentence):
    inputvec = []
    inputvec.append(" ".join(preprocess(raw_sentence)))

    ## sms_text = loadCleanTexts(config)
    ## vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), stop_words='english', strip_accents='unicode', norm='l2').fit(sms_text)
    vectorizer = loadTokenizer(config)
    inputvec = vectorizer.transform(np.array(inputvec))
    ## ogging('向量化后的稀疏矩阵', inputvec)
    return inputvec
