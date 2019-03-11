import os

from keras.applications import densenet
from keras.models import Model
from keras.layers import Activation, Dense
from keras import regularizers
from keras.optimizers import SGD

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class DenseNetTemplate:
  def __init__(self, learning_rate=0.001, num_classes=10):
        self.model_name = "DenseNet"
        self.learning_rate = learning_rate
        self.img_width = 299
        self.img_height = 299
        self.num_classes = num_classes

  def create_model(self):
      base_model = densenet.DenseNet121(input_shape=(self.img_height, self.img_width, 3),
                                     weights='imagenet',
                                     include_top=False,
                                     pooling='avg')
      for layer in base_model.layers:
        layer.trainable = True

      x = base_model.output
      x = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
      x = Activation('relu')(x)
      x = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
      x = Activation('relu')(x)
      predictions = Dense(self.num_classes, activation='softmax')(x)
      model = Model(inputs=base_model.input, outputs=predictions)
      optimizer = SGD(lr=self.learning_rate, momentum=0.9, decay=0.0, nesterov=True)
      model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    
      return model