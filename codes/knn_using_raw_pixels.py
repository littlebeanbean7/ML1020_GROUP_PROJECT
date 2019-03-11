import pandas as pd
import os
import glob
import numpy as np
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
import pickle
from sklearn.neighbors import NearestNeighbors

def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    img_folder = "../data/imgs/test/test/"
    df = pd.DataFrame(data=os.listdir("../data/imgs/test/test/"), columns=["img"])
    df = df.sort_values("img").reset_index(drop=True)
    #df = df.iloc[0:100].reset_index(drop=True)
    img_height, img_width = (40, 30)
    print("Extracting features...")
    img_features = []
    for i in tqdm(range(df.shape[0])):
        img = image.load_img(img_folder + df.iloc[i]["img"],
                             target_size=(img_height, img_width), grayscale=False)
        img_array = np.array(img).reshape(-1, )
        img_features.append(img_array)
    print("Feature extraction done!")

    print("Performing KNN...")
    neigh = NearestNeighbors(n_neighbors=11, metric="l2")
    neigh.fit(img_features)
    nn11 = []
    for i in tqdm(range(df.shape[0])):
        img = image.load_img(img_folder + df.iloc[i]["img"],
                             target_size=(img_height, img_width), grayscale=False)
        img_array = np.array(img).reshape(-1, )
        nn = neigh.kneighbors([img_array])[1].squeeze()
        nn = [df["img"].iloc[i] for i in nn]
        nn11.append(nn)
    nn11 = np.array(nn11)
    print("KNN done.")

    names = []
    for i in range(11):
        names.append("nn" + str(i))
    df_output = pd.concat([df,pd.DataFrame(nn11,columns=names)],axis=1)
    df_output.to_csv("../saved_models/nn11.csv",index=False)


if __name__ == '__main__':
    main()
