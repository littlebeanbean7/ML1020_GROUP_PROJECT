import pandas as pd
import os
import math
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    model_loaded = load_model('../saved_models/InceptionV3/bestmodel.wholedata.hdf5')
    print("model loaded.")
    img_width = 299
    img_height = 299
    img_folder = "../data/imgs/test/test/"
    df = pd.DataFrame(data=os.listdir("../data/imgs/test/test"), columns=["img"])
    #df = df.iloc[0:100].reset_index(drop=True)
    print("generating predictions for %d images" % df.shape[0])
    predict_all_data = []
    for i in range(df.shape[0]):
        if i%100 == 0:
            print("Remaining images: %d" % (df.shape[0]-i))
        img = image.load_img(img_folder + df["img"][i], target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = x / 255.
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        predict = model_loaded.predict(images).squeeze()
        predict_all_data.append(predict)
    predict_all_data = np.array(predict_all_data)
    print("predictions done.")
    df_predict = pd.DataFrame(predict_all_data, columns=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
    submission = pd.concat([df, df_predict], axis=1)
    submission.to_csv("../submissions/submission.csv", index=False)


if __name__ == '__main__':
    main()