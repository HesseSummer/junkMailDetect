# 待改进：
## 文本参数
## 多模型选择
import argparse
from utils import getModel, TOy_pred_label, logging
from config import config
from preprocess import getInputvec
import time



parser = argparse.ArgumentParser(description='一个判断输入是否为垃圾邮件的程序')
parser.add_argument('-t', '--text', required=True, help='待判断的邮件文本')
args = parser.parse_args()


def judge():
    time.sleep(0.1)
    raw_sentence = args.text
    # raw_sentence = "K tell me anything about you."
    inputvec = getInputvec(raw_sentence, config)


    clf = getModel('CNN', config)
    pred = clf.predict(inputvec)

    pred_label = TOy_pred_label(pred)

    pred_label = pred_label.tolist()
    result = "".join(pred_label)

    if result == 'ham':
        result = '正常邮件'
    else:
        result = '垃圾邮件'
    return result

if __name__ == '__main__':
    result = judge()
    print(result)
