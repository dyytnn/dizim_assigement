from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, Model 
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


def build_model(input_shape):
    googleNet_model = InceptionResNetV2(include_top = False, weights = 'imagenet', input_shape = input_shape)
    googleNet_model.trainable = True
    model = Sequential()
    model.add(googleNet_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=2, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                metrics=['accuracy'])
    return model
