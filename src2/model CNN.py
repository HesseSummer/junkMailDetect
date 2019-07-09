from keras.utils import to_categorical
from keras.models import Model
import utils
import preprocess

def trainCNN(x_train, y_train, x_val, y_val):
#构建embedding layer
    
embedding_matrix=preprocess.categorical(sms_text, sms_label)
embedding_layer = Embedding(input_dim=num_words,  # 词汇表单词数量
                            output_dim=EMBEDDING_DIM,  # 词向量维度
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)  # 词向量矩阵不进行训练

#构建、连接其他层
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')  # 占位。
embedded_sequences = embedding_layer(sequence_input)  # 返回 句子个数*50*100
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(2)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(label_num), activation='softmax')(x)
 
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.summary()


#应用模型
model.fit(x_train, y_train, batch_size=16, epochs=5, validation_data=(x_val, y_val))

#开始评估
y_pred =utils.predict(x_val,batch_size = 16)
y_val_label =utils.TOy_val_label(y_val)
y_pred_label =utils.TOy_pred_label(y_pred)
utils.show_model(y_val_label, y_pred_label)
utils.print_AUC(y_val_label, y_pred)
return model

