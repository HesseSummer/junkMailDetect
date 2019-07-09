import preprocess
from config import config
import model CNN
from utils import saveModel

if __name__ == '__main__':
    raw_data = preprocess.readData(config)
    sms_text, sms_label = preprocess.cleanData(raw_data)
    x_train, y_train, x_val, y_val ,tokenizer= preprocess.categorical(sms_text, sms_label)

    clf = model CNN.trainCNN(x_train, y_train, x_val, y_val)
    saveModel(clf, "CNN", config)
 