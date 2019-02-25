import os
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import SGD

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class InceptionV3ClassifierTemplate:
    def __init__(self, learning_rate=0.0001, num_classes=10):
        self.model_name = "InceptionV3"
        self.learning_rate = learning_rate
        self.img_width = 299
        self.img_height = 299
        self.num_classes = num_classes

    def create_model(self):
        InceptionV3_notop = InceptionV3(include_top=False,
                                        weights='imagenet', input_tensor=None,
                                        input_shape=(self.img_height, self.img_width, 3))
        output = InceptionV3_notop.get_layer(index=-1).output  # Shape: (8, 8, 2048)
        output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
        output = Flatten(name='flatten')(output)
        output = Dense(self.num_classes, activation='softmax', name='predictions')(output)
        InceptionV3_model = Model(InceptionV3_notop.input, output)
        optimizer = SGD(lr=self.learning_rate, momentum=0.9, decay=0.0, nesterov=True)
        InceptionV3_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return InceptionV3_model
