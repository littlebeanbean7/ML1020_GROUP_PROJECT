import pandas as pd
import os
import glob
import numpy as np
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
import pickle
from sklearn.neighbors import NearestNeighbors
from keras.models import load_model
from keras.models import Model

saved_feature_file = "../saved_models/features_from_inceptionV3_for_knn.pickle"
def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    img_folder = "../data/imgs/test/test/"
    df = pd.DataFrame(data=os.listdir("../data/imgs/test/test/"), columns=["img"])
    df = df.sort_values("img").reset_index(drop=True)
    #df = df.iloc[0:100].reset_index(drop=True)
    img_width = 299
    img_height = 299
    model_file = "../saved_models/InceptionV3/bestmodel.wholedata.hdf5"
    model_loaded = load_model(model_file)
    output = model_loaded.get_layer(name="flatten").output
    model_loaded = Model(model_loaded.input, output)

    if saved_feature_file is None:
        print("Extracting features...")
        img_features = []
        for i in tqdm(range(df.shape[0])):
            img = image.load_img(img_folder + df["img"][i],
                                 target_size=(img_width, img_height))
            x = np.array(img)
            x = x / 225.0
            x = np.expand_dims(x, axis=0)
            x = np.vstack([x])
            x = preprocess_input(x)
            predict = model_loaded.predict(x)
            predict = predict.reshape(-1, )
            img_features.append(predict)
        print("Feature extraction done!")
        print("Trying to save the features...")
        with open("../saved_models/features_from_inceptionV3_for_knn.pickle", 'wb') as handle:
            pickle.dump(img_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saving features done.")
    else:
        with open(saved_feature_file, 'rb') as handle:
            img_features = pickle.load(handle)

    print("Performing KNN...")
    neigh = NearestNeighbors(n_neighbors=11, metric="l2")
    neigh.fit(img_features)
    nn11 = []
    for i in tqdm(range(df.shape[0])):
        img = image.load_img(img_folder + df["img"][i],
                             target_size=(img_width, img_height))
        x = np.array(img)
        x = x/255.0
        x = np.expand_dims(x, axis=0)
        x = np.vstack([x])
        x = preprocess_input(x)
        predict = model_loaded.predict(x)
        predict = predict.reshape(-1, )
        nn = neigh.kneighbors([predict])[1].squeeze()
        nn = [df["img"].iloc[i] for i in nn]
        nn11.append(nn)
    nn11 = np.array(nn11)
    print("KNN done.")

    names = []
    for i in range(11):
        names.append("nn" + str(i))
    df_output = pd.concat([df,pd.DataFrame(nn11,columns=names)],axis=1)
    df_output.to_csv("../saved_models/nn11_inceptionv3.csv",index=False)


if __name__ == '__main__':
    main()
