# CS470 Final Project
## Aiden Seay - Spring 2024

### Problem
This program running in a Jupyter Notebook uses different ML algorithms classifying whether or not an email is spam. 

### About the dataset
The dataset can be found in the `data` directory

You can find the dataset [here]. (https://www.kaggle.com/datasets/colormap/spambase/data).

The dataset will be split by the following
* 

### Algorithms Implemented
All algorithms can be found in the `Utilities` directory

Each algorithm implemented is below:
* Naive Bayes
* Logistic Regression
* KNN

### Measuring Performance
Performance will be measured by the following:

* Accuracy (acc): The ratio of correctly predicted observation to the total observations.
* False Positive (fp): The number of negative instances wrongly predicted as positive.
* True Positive (tp): The number of positive instances correctly predicted as positive.
* Area Under the Curve (AUC): The area under the Receiver Operating Characteristic (ROC) curve, which shows the trade-off between sensitivity (true positive rate) and specificity (true negative rate).