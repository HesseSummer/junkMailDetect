import utils
import CNN

def trainCNN(x_train, y_train, x_val, y_val, embedding_matrix):
    clf=CNN.CNN(embedding_matrix).fit(x_train, y_train, batch_size=16, epochs=5, validation_data=(x_val, y_val))
    y_pred =utils.predict(x_val,batch_size = 16)
    y_val_label =utils.TOy_val_label(y_val)
    y_pred_label =utils.TOy_pred_label(y_pred)
    utils.show_Model(y_val_label, y_pred_label,"卷积神经网络")
    utils.print_AUC(y_val_label, y_pred)
    return clf

