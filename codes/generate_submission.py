import pandas as pd
import os
import glob
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator

aug = True

def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    img_width = 299
    img_height = 299
    img_folder = "../data/imgs/test/test/"
    df = pd.DataFrame(data=os.listdir("../data/imgs/test/test"), columns=["img"])
    df.sort_values("img",inplace=True)
    #df = df.iloc[0:2].reset_index(drop=True)
    print("generating predictions for %d images" % df.shape[0])

    gen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    model_saved_dir = "../saved_models/InceptionV3/"
    model_files_list = list(glob.glob(model_saved_dir + "*hdf5*"))
    model_files_list.sort(reverse=True)
    for file in model_files_list:
        model_loaded = load_model(file)
        file = file.split("/")[-1]
        print("%s model loaded." % file)
        predict_all_data = []
        for i in tqdm(range(df.shape[0])):
            img = image.load_img(img_folder + df["img"][i], target_size=(img_width, img_height))
            x = np.array(img)
            if aug == False:
                x = x / 255.
                x = np.expand_dims(x, axis=0)
                x = np.vstack([x])
                predict = model_loaded.predict(x).squeeze()
            else:
                predictitions_w_aug = []
                for j in range(5):
                    seed = 1234
                    trans_img = gen.random_transform(x, seed + j + i)
                    trans_img = trans_img/255.
                    trans_img = np.expand_dims(trans_img, axis=0)
                    trans_img = np.vstack([trans_img])
                    predict = model_loaded.predict(trans_img).squeeze()
                    predictitions_w_aug.append(predict)
                predictitions_w_aug = np.array(predictitions_w_aug)
                predict = np.average(predictitions_w_aug, axis=0)
            predict_all_data.append(predict)
        predict_all_data = np.array(predict_all_data)
        print("\npredictions done with model %s" % file)
        df_predict = pd.DataFrame(predict_all_data, columns=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
        submission = pd.concat([df, df_predict], axis=1)
        submission.to_csv("../submissions/" + file + ".submission.csv"+".aug"+str(aug), index=False)
    print("Done.")

if __name__ == '__main__':
    main()