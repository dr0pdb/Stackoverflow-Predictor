# Stackoverflow Predictor

![](stackoverflow_logo.png)

Stackoverflow Predictor is a Machine Learning driven full stack web application which aims at classifying a **Stackoverflow** question into closed or open depending on the various features such as ```Title```, ```Body```, ```Reputation of User``` etc.

It is my first attempt at trying out various algorithms popular in the field of **Machine learning** and **Natural Language Processing**. As a result of this, the main focus has been on the basics and the fundamental algorithms which are easy to understand and implement using the various tools and libraries at the disposal.

## Dataset
The Project was inspired from a past **Kaggle** competition aimed at predicting whether a new **Stackoverflow** question will be closed by the community. The Dataset and the detailed problem description can be found [here](https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow).

## Results
Since the computation overload was huge, I used only the 25,000 entries of the dataset in order to train the classifier using the various classification algorithms.

Following is the accuracy obtained on those algorithms:
1. Random forest classifier - 75%
2. Logistic Regression - 73% 
3. Kernel SVM - 70%
4. Naive Bayes - 71%

There is definitely scope for improvement and after exploring the field even more I would try to improve the classifier further.

## Libraries used
1. Scikit-learn
2. NLTK
3. Tensorflow
4. Pandas
5. Numpy
6. Django
7. Django-Bootstrap3
8. Tickle

## Modules required
The ```REQUIREMENTS.txt``` file can be used to install all the required modules instantly.

## Running
A simple command ```python manage.py runserver``` can be used after installing the required modules to run the application

## Screenshots
![](/screenshots/Form.png)

![](/screenshots/Negative_Response.png)

## Contributing
PRs are most welcome since i am extremely new to Machine Learning as well as Web designing.

There are two branches in the project:
1. master : It contains the most stable build of the application.
2. development : It contains the most updated and unstable version of the application.

All your pull requests are supposed to be directed at the ```development``` branch.
