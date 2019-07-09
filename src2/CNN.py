from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
import preprocess

def CNN(embedding_matrix):
    embedding_matrix=preprocess.train_dic(sms_text)
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
    return model