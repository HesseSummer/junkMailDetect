import preprocess
from config import config
import models
from utils import saveModel, logging

if __name__ == '__main__':
    raw_data = preprocess.readData(config)
    sms_text, sms_label = preprocess.cleanData(raw_data)

    x_train, y_train, x_test, y_test = preprocess.vectorize(sms_text, sms_label, config)

    clf = models.trainNB(x_train, y_train, x_test, y_test)
    saveModel(clf, "贝叶斯", config)
    clf = models.trainTree(x_train, y_train, x_test, y_test)
    saveModel(clf, "决策树", config)
    clf = models.trainRandomForest(x_train, y_train, x_test, y_test)
    saveModel(clf, "随机森林", config)
    clf = models.trainLinearSVC(x_train, y_train, x_test, y_test)
    saveModel(clf, "线性SVC", config)
    clf = models.trainSGDSVM(x_train, y_train, x_test, y_test)
    saveModel(clf, "SGDSVM", config)
    clf = models.trainSGDLog(x_train, y_train, x_test, y_test)
    saveModel(clf, "SDD逻辑回归", config)


