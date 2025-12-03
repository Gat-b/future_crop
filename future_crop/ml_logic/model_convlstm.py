from keras import Model, Sequential, layers, regularizers, optimizers
from keras.layers import Input, ConvLSTM3D
from keras.callbacks import EarlyStopping

def convlstm_model(X) :
    '''
    Docstring pour convlstm_model

    :param X_shape: Description
    '''

    model = Sequential()
    model.add(Input(X.shape[1:]))

    # Conv Layer
    model.add((ConvLSTM3D(kernel= ,
                          )))
