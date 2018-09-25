from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Merge, LSTM, Dense
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras import regularizers
from dataHelper import *
import time
import pandas as pd

def trainOrLoad(flag = 1):
    timesteps = 63
    data_dim = 200
    nb_classes = 7

    lstm_model = Sequential()
    lstm_model.add(LSTM(32, input_shape=(timesteps, data_dim)))
    # model.add(Dropout(0.5))
    lstm_model.add(Dense(16, activation='relu'))
    lstm_model.add(Dense(nb_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))

    lstm_model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

    x_train_feature,y_train,x_test_feature,y_test = main()
    y_train = to_categorical(y_train, nb_classes)
    # y_test = to_categorical(y_test, 7)
    if flag:
        time1 = time.time()
        lstm_model.fit(x_train_feature, y_train,
                    batch_size=64, nb_epoch=100
                    # validation_split=0.1
                    )

        json_string = lstm_model.to_json()
        open('./ques_models/lstm_model/lstm_model2.json','w').write(json_string)
        lstm_model.save_weights('./ques_models/lstm_model/lstm_model2.h5')

        # lstm_model.predict_classes()
        y_pre = lstm_model.predict_classes(x_test_feature)
        y_test = [int(x.strip()) for x in y_test]
        # print(y_test)
        # print(y_pre)
        time2 = time.time()
        print("耗时%d"%(time2-time1))
        print(classification_report(y_test,y_pre))

    else:
        lstm_model = model_from_json(open('./ques_models/lstm_model/lstm_model2.json').read())
        lstm_model.load_weights('./ques_models/lstm_model/lstm_model2.h5')
        y_test = [int(x.strip()) for x in y_test]
        y_pre = lstm_model.predict_classes(x_test_feature)
        print(y_pre)
        ques_result = pd.DataFrame({'y_test': y_test, 'y_pre': y_pre})
        ques_result.to_csv('./results/submission/ques_result.txt', sep='\t', encoding='utf-8', index=None)
        print()
        print(classification_report(y_test,y_pre))
        print("Accuracy:", accuracy_score(y_test, y_pre))

trainOrLoad(flag=1)
