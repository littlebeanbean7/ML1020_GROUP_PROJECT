# ML1020_GROUP_PROJECT
State Farm Distracted Driver Detection Competition   
https://www.kaggle.com/c/state-farm-distracted-driver-detection   

Final standing:   
Private Leaderboard score 0.26251   
152 out of 1440 teams (Top 10.6%)   

Public Leaderboard score 0.32938   
216 out of 1440 teams (Top 15.0%)   

## What's included in this repository

### write-up/
Presentation slides, final submitted report, and mid-project reports are located here. Reader are suggest to start from the final submitted report and presentation slides.   

### codes/    
* InceptionV3ClassifierTemplate.py, codes/ResNetClassifierTemplate.py  
Template class that generate a network with pretrained weights downloaded from Keras, for InceptionV3 and ResNet50 respectively.   

* Classifier.py    
This is the top level code (main function) where the training, cross validation of models take place. Use this code to train models. It will take a downloaded weights from either InceptionV3ClassifierTemplate.py or ResNetClassifierTemplate.py, and train 5 models during cross validation and 1 model trained on all avaiable training data. It can also produce cross-validation score (loss and accuracy) on the respective models.   

* knn_using_inceptionv3.py   
Find K-nearest neighbors using the outputs of last layers of InveptionV3 model.   

* knn_using_raw_pixels.py   
Find K-nearest neighbors using the pixel values of reduced-size images.

* generate_submission.py   
Codes to generate submissions for each of models trained in Classifier.py. Image augementations are performend on the test images.    

* generate_submission_knn.py   
Collect K-nearest neighbors and perfome averaging based on the predictions of the neighbors.

* knn_accuracy_trainset.ipynb   
Notebook to evaluate KNN's accuracy on training set.   

* average_submission.ipynb   
Notebook to generate the final submissions.   
