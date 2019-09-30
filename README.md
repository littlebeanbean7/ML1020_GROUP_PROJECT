# Kaggle Competition: Image Classification on Distracteds Drivers
State Farm Distracted Driver Detection Competition   
https://www.kaggle.com/c/state-farm-distracted-driver-detection   

Final standing:   
Private Leaderboard score 0.20022   
63 out of 1440 teams (Top 4.4%)   

Public Leaderboard score 0.21994   
103 out of 1440 teams (Top 7.2%)   


## What's included in this repository

### write-up/
Presentation slides, final submitted report, and mid-project reports are located here. Readers are suggested to start from the final submitted report and presentation slides.  

### submissions/   
* final_candidate1.csv   
Final submissions to kaggle - candidate1
* final_candidate2.csv   
Final submissions to kaggle - candidate2   
For description of candidate1 and candidate2, please refer to presentation slides.   

### codes/    
* InceptionV3ClassifierTemplate.py, codes/ResNetClassifierTemplate.py  
Template class that generate a network with pretrained weights downloaded from Keras, for InceptionV3 and ResNet50 respectively.   

* Classifier.py    
This is the top level code (main function) where the training, cross-validation of models take place. Use this code to train models. It will take downloaded weights from either InceptionV3ClassifierTemplate.py or ResNetClassifierTemplate.py, and train 5 models during cross-validation and 1 model trained on all available training data. It can also produce a cross-validation score (loss and accuracy) on the respective models.   

* knn_using_inceptionv3.py   
Find K-nearest neighbors using the outputs of last layers of InveptionV3 model.   

* knn_using_raw_pixels.py   
Find K-nearest neighbors using the pixel values of reduced-size images.

* generate_submission.py   
Codes to generate submissions for each of the models trained in Classifier.py. Image augmentations are performed on the test images.    

* generate_submission_knn.py   
Collect K-nearest neighbors and perform averaging based on the predictions of the neighbors.

* knn_accuracy_trainset.ipynb   
Notebook to evaluate KNN's accuracy on the training set.   

* average_submission.ipynb   
Notebook to generate the final submissions.   

## What's not included in this repository  
The resulting models and intermediate submission files are stored in saved_model/ and submissions/ respectively, during the project. However, due to the size of the models, they are uploaded to GitHub. Intermediate submission files are not uploaded to GitHub either.   

To reproduce the results presented in the final report and presentation slides, you will have to retrain all the models and generate the submission files or contact the owner of this repository.   
