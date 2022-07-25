# Heart Disease Prediction Using Machine Learning and Streamlit Deployment
 
## Project Description
 According to World Health Organisation (WHO), every year around 17.9 million deaths are due to cardiovascular diseases (CVDs) predisposing CVD becoming the leading cause of death globally. CVDs are a group of disorders of the heart and blood vessels, if left untreated it may cause heart attack. Heart attack occurs due to the presence of obstruction of blood flow into the heart. The presence of blockage may be due to the accumulation of fat, cholesterol, and other substances. Despite treatment has improved over the years and most CVD’s pathophysiology have been elucidated, heart attack can still be fatal.

 Thus, clinicians believe that prevention of heart attack is always better than curing it. After many years of research, scientists and clinicians discovered that, the probability of one’s getting heart attack can be determined by analysing the patient’s age, gender, exercise induced angina, number of major vessels, chest pain indication, resting blood pressure, cholesterol level, fasting blood sugar, resting electrocardiographic results, and maximum heart rate achieved.

The objective of this project is to predict the possibility of one getting heart heart attack using machine learning. This project also aims to select the best model which the highest accuracy in predicting the possibility of one getting heart attack.

## Running the Project
This model run using Python programming and the libraries available. It is also deployed using Streamlit for end user to predict their probability of getting heart attack. The model also tested with a set new dataset to calculate the accuracy of the selected model.

## Project Insight
To achieve the objective of this project, machine learning approach is used considering the dataset availabel. This project used pipelines to predict which machine learning approach is the best suited for this dataset. The machine learning approach used are Logistic Regression, Decision Tree, Random Forest, Gradient Boosting(gboost) and K Nearest Neighbor. Hyperparameter tuning is applied to the model that give the highest score. 

The selected model is then tested with new dataset to determine the accuracy of the model with another dataset. Testing the model also important to verify the usability of the model with another data. 

## Accuracy
After cleaning, selecting the best features and training the data, this model achive up to 0.8 accuracy. 

###### Image shows the training classification report.
![Classification report](https://github.com/noorhanifah/Bank-Marketing-Campaign-Analysis/blob/main/Image/Classification_report.PNG)

###### Graph shows the training and validation of the accuracy 
![Plotted Accuracy](https://github.com/noorhanifah/Bank-Marketing-Campaign-Analysis/blob/main/Image/Plotted%20Accuracy.PNG)

###### Training and validation of the accuracy shows on TensorBoard 
![Training Accuracy](https://github.com/noorhanifah/Bank-Marketing-Campaign-Analysis/blob/main/Image/Accuracy.PNG)

###### Graph shows the training loss and validation loss of the model
![Plotted Loss](https://github.com/noorhanifah/Bank-Marketing-Campaign-Analysis/blob/main/Image/Plotted_loss.PNG)

###### Training and validation of the loss shows on TensorBoard 
![Training Loss](https://github.com/noorhanifah/Bank-Marketing-Campaign-Analysis/blob/main/Image/Loss.PNG)

## A little discussion

## Build With
 ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
 ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
 ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
 ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
 ![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
 ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
 ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

## Credit
The dataset can be downloaded from Kaggle dataset at https://www.kaggle.com/datasets/kunalgupta2616/hackerearth-customer-segmentation
