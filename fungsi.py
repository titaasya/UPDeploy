import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, LeakyReLU


def make_model():
    base_model_1 = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet',
                                                     input_shape=(32, 32, 3), classes=10)
    # Lets add the final layers to these base models where the actual classification is done in the dense layers
    model = Sequential()
    # Adds the base model (in this case vgg19 to model_1)
    model.add(base_model_1)
    # Since the output before the flatten layer is a matrix we have to use this function to get a vector of the form nX1 to feed it into the fully connected layers
    model.add(Flatten())
    # Add the Dense layers along with activation and batch normalization
    model.add(Dense(1024, activation=('relu'), input_dim=512))
    model.add(Dense(512, activation=('relu')))
    model.add(Dense(256, activation=('relu')))
    # model_1.add(Dropout(.3))#Adding a dropout layer that will randomly drop 30% of the weights
    model.add(Dense(128, activation=('relu')))
    # model_1.add(Dropout(.2))
    # # This is the classification layer
    model.add(Dense(10, activation=('softmax')))
    return model
