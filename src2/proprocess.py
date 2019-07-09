import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocess, logging

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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

def tokenize(sms_text):
    MAX_NUM_WORDS = 2000
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS) ## 最终选取频率前MAX_NUM_WORDS个单词
    tokenizer.fit_on_texts(sms_text)
    return tokenizer

def categorical(sms_text, sms_label):
    MAX_SEQUENCE_LENGTH = 50 ## 长度超过MAX_SEQUENCE_LENGTH则截断，不足则补0
    tokenizer=tokenize(sms_text)
    sequences = tokenizer.texts_to_sequences(sms_text) ## 是一个二维数值数组，每一个数值都是对应句子对应单词的**索引**
    dataset = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) 
    labels = to_categorical(np.asarray(sms_label)) ## 将label转为独热编码形式

    """打乱"""
    indices = np.arange(dataset.shape[0])
    np.random.shuffle(indices)
    dataset = dataset[indices]
    labels = labels[indices]

    """划分"""
    size_dataset = len(dataset)
    size_trainset = int(round(size_dataset*0.7))
    x_train = dataset[0:size_trainset]
    y_train = labels[0:size_trainset]

    x_val = dataset[size_trainset+1: size_dataset]
    y_val = labels[size_trainset+1: size_dataset]
    return x_train, y_train, x_val, y_val
    
def train_dic(sms_text):
    MAX_NUM_WORDS = 2000
    tokenizer=tokenize(sms_text)
    embedding_dic = {}
    file_path = '../glove/glove.6B.100d.txt'

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_dic[word] = coefs
            
            """选取部分单词"""
    word_index = tokenizer.word_index ## 得到一个字典，key是选择的单词，value是它的索引

    logging("共有{}个单词，示例：",format(len(word_index)))
    logging(list(word_index.keys())[0:5], list(word_index.values())[0:5]) 

    """准备这些单词的embedding_matrix"""
    EMBEDDING_DIM = 100 ## 令词向量的维度是100
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1) ## 为什么要加一？
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embedding_dic.get(word)
        if embedding_vector is not None: ## 单词在emmbeding_dic中存在时
            embedding_matrix[i] = embedding_vector
    return embedding_matrix