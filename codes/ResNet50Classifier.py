import pandas as pd
import numpy as np
import os
import math

from keras.models import load_model
from sklearn.model_selection import StratifiedKFold

from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import argparse
import pickle
import glob

parser = argparse.ArgumentParser()
#parser.add_argument("-m", "--modelname", required=True,
#                    help="Model Name (should be ResNet50 for this file)")
parser.add_argument("-ds", "--downsample", default=0, type=bool,
                    help="Whether to downsample dataset")
parser.add_argument('-lr', "--learning_rate", default=0.0001, type=float)
parser.add_argument('-epochs', "--nbr_epochs", default=5, type=int)
parser.add_argument('-bs', "--batch_size", default=32, type=int)
args = parser.parse_args()

learning_rate = 0.0001
nbr_epochs = 5
batch_size = 32

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ResNet50Classifier:
    def __init__(self, train_imgs_csvfile="../data/driver_imgs_list.csv"):
        self.df = pd.read_csv(train_imgs_csvfile)
        self.classnames = np.unique(self.df["classname"])
        if args.downsample is True:
            self.df = self.df.iloc[list(range(0, self.df.shape[0], 200))]
        self.model_name = "ResNet50"
        self.learning_rate = args.learning_rate
        self.nbr_epochs = args.nbr_epochs
        self.batch_size = args.batch_size
        self.img_width = 224
        self.img_height = 224

    def fit(self, saved_folder=None):
        if saved_folder is None:
            print("Model saved path is not provided.")
            raise ValueError
        if os.path.isdir(saved_folder) is False:
            os.mkdir(saved_folder)
        if os.path.isdir(saved_folder+"/"+self.model_name) is False:
            os.mkdir(saved_folder+"/"+self.model_name)

        scores = {}
        scores["train_loss"] = []
        scores["train_acc"] = []

        df = self.df[["img", "classname"]]
        y = df["classname"]
        ResNet50_notop = ResNet50(include_top=False, weights='imagenet',
                                        input_tensor=None,
                                        input_shape=(self.img_height, self.img_width, 3))

        output = ResNet50_notop.get_layer(index=-1).output  # Shape: (7, 7, 2048)
        output = AveragePooling2D((7, 7), strides=(7, 7), name='avg_pool')(output)
        output = Flatten(name='flatten')(output)
        output = Dense(len(self.classnames), activation='softmax', name='predictions')(output)

        ResNet50_model = Model(ResNet50_notop.input, output)
        optimizer = SGD(lr=self.learning_rate, momentum=0.9, decay=0.0, nesterov=True)

        ResNet50_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.1,
                zoom_range=0.1,
                rotation_range=10.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True)

        train_generator = train_datagen.flow_from_dataframe(
                df,
                directory="../data/imgs/train/", x_col="img", y_col="classname",
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                shuffle=True,
                class_mode='categorical')
        history = ResNet50_model.fit_generator(
                train_generator,
                steps_per_epoch=math.ceil(df.shape[0] / self.batch_size),
                nb_epoch=self.nbr_epochs)
        saved_model_file = saved_folder + "/" + self.model_name + "/bestmodel.wholedata.hdf5"
        ResNet50_model.save(saved_model_file)
        with open(saved_folder + "/" + self.model_name + "/history.wholedata" + ".pickle", 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        train_res = ResNet50_model.evaluate_generator(train_generator,
                                   steps=math.ceil(df.shape[0] / self.batch_size))
        scores["train_loss"].append(train_res[0])
        scores["train_acc"].append(train_res[1])
        # Summmary
        df_scores = pd.DataFrame(scores)
        if os.path.isdir(saved_folder) is False:
            os.mkdir(saved_folder)
        with open(saved_folder + "/" + self.model_name + "/wholedata_score.pickle", 'wb') as handle:
            pickle.dump(df_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(saved_folder + "/" + self.model_name + "/wholedata_score.txt", 'w') as handle:
            handle.write(saved_folder + "\n")
            handle.write(str(df_scores))
            handle.write("\n")
        return df_scores

    def predict(self, X):
        pred_prob = np.zeros((len(X),))
        return pred_prob

    def cross_validate(self, k=5, saved_folder=None):
        if saved_folder is None:
            print("Model saved path is not provided.")
            raise ValueError
        if os.path.isdir(saved_folder) is False:
            os.mkdir(saved_folder)
        if os.path.isdir(saved_folder+"/"+self.model_name) is False:
            os.mkdir(saved_folder+"/"+self.model_name)

        scores = {}
        scores["train_loss"] = []
        scores["train_acc"] = []
        scores["val_loss"] = []
        scores["val_acc"] = []

        df = self.df[["img", "classname"]]
        y = df["classname"]
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        for i, (train_index, val_index) in enumerate(skf.split(df, y)):
            print("CV round %d..." % i)
            df_train = df.iloc[train_index].reset_index()
            df_val = df.iloc[val_index].reset_index()
            ResNet50_notop = ResNet50(include_top=False, weights='imagenet',
                                            input_tensor=None,
                                            input_shape=(self.img_height, self.img_width, 3))
            output = ResNet50_notop.get_layer(index=-1).output  # Shape: (7, 7, 2048)
            output = AveragePooling2D((7, 7), strides=(7, 7), name='avg_pool')(output)
            output = Flatten(name='flatten')(output)
            output = Dense(len(self.classnames), activation='softmax', name='predictions')(output)
            ResNet50_model = Model(ResNet50_notop.input, output)
            # ResNet50_model.summary()
            optimizer = SGD(lr=self.learning_rate, momentum=0.9, decay=0.0, nesterov=True)
            ResNet50_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            # autosave best Model
            best_model_file = saved_folder+"/"+self.model_name + "/bestmodel.hdf5.cv" + str(i)
            best_model_callback = ModelCheckpoint(best_model_file,
                                                  monitor='val_loss', verbose=1, save_best_only=True)
            early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
            callbacks_list = [best_model_callback,early]
            # this is the augmentation configuration we will use for training
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.1,
                zoom_range=0.1,
                rotation_range=10.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True)
            # this is the augmentation configuration we will use for validation:
            # only rescaling
            val_datagen = ImageDataGenerator(rescale=1. / 255)
            train_generator = train_datagen.flow_from_dataframe(
                df_train,
                directory="../data/imgs/train/", x_col="img", y_col="classname",
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                shuffle=True,
                # save_to_dir = '../data/aug_output/',
                # save_prefix = 'aug',
                # classes = class_names,
                class_mode='categorical')

            validation_generator = val_datagen.flow_from_dataframe(
                df_val,
                directory="../data/imgs/train/", x_col="img", y_col="classname",
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                shuffle=True,
                # save_to_dir = '../data/aug_output/',
                # save_prefix = 'aug',
                # classes = class_names,
                class_mode='categorical')

            history = ResNet50_model.fit_generator(
                train_generator,
                steps_per_epoch=math.ceil(df_train.shape[0] / self.batch_size),
                nb_epoch=self.nbr_epochs,
                validation_data=validation_generator,
                validation_steps=df_val.shape[0] / self.batch_size,
                callbacks=callbacks_list)
            with open(saved_folder + "/" + self.model_name +
                      "/history.cv" + str(i) + ".pickle", 'wb') as handle:
                pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #print(history)
            val_res = ResNet50_model.evaluate_generator(validation_generator, \
                                                            steps=math.ceil(df_val.shape[0] / self.batch_size))
            train_res = ResNet50_model.evaluate_generator(train_generator, \
                                                             steps=math.ceil(df_train.shape[0] / self.batch_size))
            scores["train_loss"].append(train_res[0])
            scores["train_acc"].append(train_res[1])
            scores["val_loss"].append(val_res[0])
            scores["val_acc"].append(val_res[1])
        # Summmary
        df_scores = pd.DataFrame(scores)
        df_scores.index.name = "CV round"
        df_scores = df_scores.T
        df_scores["mean"] = df_scores.mean(axis=1)
        df_scores["std"] = df_scores.std(axis=1)
        if saved_folder is None:
            return df_scores
        if os.path.isdir(saved_folder) is False:
            os.mkdir(saved_folder)
        with open(saved_folder+"/"+self.model_name+"/cv_score.pickle", 'wb') as handle:
            pickle.dump(df_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(saved_folder+"/"+self.model_name+"/cv_score.txt", 'w') as handle:
            handle.write(saved_folder+"\n")
            handle.write(str(df_scores))
            handle.write("\n")
        return df_scores

    def load_models(self, saved_folder):
        print("loading models from " + saved_folder+"/"+self.model_name)
        if os.path.isdir(saved_folder+"/"+self.model_name) is True:
            files = sorted(glob.glob(saved_folder+"/"+self.model_name+"/*"))
            for f in files:
                pass
        else:
            print(saved_folder+"/"+self.model_name + " is empty!")
            raise ValueError

def main():
    clf = ResNet50Classifier()
    df = clf.cross_validate(5, saved_folder="../saved_models/")
    df = clf.fit(saved_folder="../saved_models/")
    print(df)

if __name__ == '__main__':
    main()
