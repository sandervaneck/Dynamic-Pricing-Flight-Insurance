import numpy as np
from imblearn.metrics import specificity_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score, f1_score, precision_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

def total_score(auc, accuracy, precision, recall, f1, specifity, rmse):
    print("auc")
    print(auc)
    print("accuracy")
    print(accuracy)
    print("precision")
    print(precision)
    print("recall")
    print(recall)
    print("f1")
    print(f1)
    print("specifity")
    print(specifity)
    print("rmse")
    print(rmse)
    return (auc*0.1+accuracy*0.05+precision*0.15+recall*0.45+f1*0.1+specifity*0.05+(1-rmse)*0.1)