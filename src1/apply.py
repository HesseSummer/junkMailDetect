# 待改进：
## 文本参数
## 多模型选择
import argparse
from utils import getModel, logging
from config import config
from preprocess import getInputvec
import time



parser = argparse.ArgumentParser(description='一个判断输入是否为垃圾邮件的程序')
parser.add_argument('-t', '--text', required=True, help='待判断的邮件文本')
args = parser.parse_args()


def judge():
    time.sleep(0.1)
    raw_sentence = args.text

    inputvec = getInputvec(raw_sentence)

    clf = getModel('线性SVC', config)
    pred = clf.predict(inputvec)
    pred = ''.join(pred)
    if pred == 'ham':
        result = '正常邮件'
    else:
        result = '垃圾邮件'
    return result

if __name__ == '__main__':
    result = judge()
    print(result)

