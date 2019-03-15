import os
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class ResNet50ClassifierTemplate:
    def __init__(self, learning_rate=0.0001, num_classes=10):
        self.model_name = "ResNet50"
        self.learning_rate = learning_rate
        self.img_width = 224
        self.img_height = 224
        self.num_classes = num_classes

    def create_model(self):
        ResNet50_notop = ResNet50(include_top=False,
                                        weights='imagenet', input_tensor=None,
                                        input_shape=(self.img_height, self.img_width, 3))
        output = ResNet50_notop.get_layer(index=-1).output  # Shape: (7, 7, 2048)
        output = AveragePooling2D((7, 7), strides=(7, 7), name='avg_pool')(output)
        output = Flatten(name='flatten')(output)
        output = Dense(self.num_classes, activation='softmax', name='predictions')(output)
        ResNet50_model = Model(ResNet50_notop.input, output)
        optimizer = Adam()
        ResNet50_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return ResNet50_model
