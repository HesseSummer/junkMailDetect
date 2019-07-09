import preprocess
from config import config
import modelCNN
from utils import saveModel

if __name__ == '__main__':
    raw_data = preprocess.readData(config)
    sms_text, sms_label = preprocess.cleanData(raw_data)
    x_train, y_train, x_val, y_val= preprocess.categorical(sms_text, sms_label)
    embedding_layer=preprocess.train_dic(sms_text)
    clf = modelCNN.trainCNN(x_train, y_train, x_val, y_val, embedding_layer)
    saveModel(clf, "CNN", config)
 