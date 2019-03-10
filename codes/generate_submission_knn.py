import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm


#knn_model_file = "../saved_models/nn11_inceptionv3.csv"
#submission_file = "../submissions/ensemble_5folds_and_fulldata_augTrue_inceptionv3knn.csv"

model_name = "Resnet50"
knn_model_file = "../saved_models/nn11_rawpixels.csv"
submission_file = "../submissions/" + model_name + "/ensemble_5folds_and_fulldata_augTrue_rawpixelknn.csv"
base_submissino_file = "../submissions/" + model_name + "/ensemble_ResNet_augTrue_fivecv_whole.csv"
columns=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']

def main():
    df_knn = pd.read_csv(knn_model_file)
    df_knn = df_knn[["img", "nn0", "nn1", "nn2", "nn3", "nn4"]]
    df_knn = df_knn.set_index("img")

    #df is the ensemble of 5cv models and whole data model + image augmentation on test set
    df = pd.read_csv(base_submissino_file)
    df = df.sort_values("img").reset_index(drop=True)
    df = df.set_index("img")

    sub_w_knn = pd.DataFrame(columns=["img"] + columns)
    for i in tqdm(range(df.shape[0])):
        target_img = df.iloc[i].name
        nns = df_knn.loc[target_img]
        scores = df.loc[target_img]
        for n in nns:
            scores = scores + df.loc[n]
        scores = scores / (len(nns)+1)
        scores["img"] = target_img
        scores = pd.DataFrame([scores])
        scores = scores[["img"] + columns]
        sub_w_knn = pd.concat([sub_w_knn, scores], axis=0)
        # break
    sub_w_knn.to_csv(submission_file, index=False)


if __name__ == '__main__':
    main()