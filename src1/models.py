from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
import utils


def trainNB(x_train, y_train, x_test, y_test):
    clf = MultinomialNB().fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    utils.showModel(y_test, y_pred, "朴素贝叶斯")
    utils.printAUC(clf, x_test, y_test)
    return clf


def trainTree(x_train, y_train, x_test, y_test):
    clf = tree.DecisionTreeClassifier().fit(x_train.toarray(), y_train)
    y_pred = clf.predict(x_test.toarray())
    utils.showModel(y_test, y_pred, "决策树")
    utils.printAUC(clf, x_test, y_test)
    return clf


def trainRandomForest(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    utils.showModel(y_test, y_pred, "随机森林")
    utils.printAUC(clf, x_test, y_test)
    return clf


def trainLinearSVC(x_train, y_train, x_test, y_test):
    clf = LinearSVC().fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    utils.showModel(y_test, y_pred, "线性SVC")
    utils.printAUCsvm(clf, x_test, y_test)
    return clf


def trainSGDSVM(x_train, y_train, x_test, y_test):
    clf = SGDClassifier(alpha=0.0001, n_iter=50).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    utils.showModel(y_test, y_pred, "SGDSVM")
    utils.printAUCsvm(clf, x_test, y_test)
    return clf


def trainSGDLog(x_train, y_train, x_test, y_test):
    clf = SGDClassifier(loss='log', alpha=0.0001, n_iter=50).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    utils.showModel(y_test, y_pred, "SGD逻辑回归")
    utils.printAUC(clf, x_test, y_test)
    return clf

