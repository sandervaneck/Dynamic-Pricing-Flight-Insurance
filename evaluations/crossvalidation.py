import numpy as np
import pandas as pd
from imblearn.metrics import specificity_score
from sklearn.metrics import make_scorer, precision_score, log_loss, confusion_matrix, roc_auc_score, accuracy_score, \
    recall_score, f1_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from statsmodels.tools.eval_measures import rmse

from excelWriters.plotter import estimate_cost


def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity
def crossvalscore_glm(glm_result, test_X_c, y, title):

    predictions = glm_result.predict(test_X_c)
    # Assuming you have a binary classification problem, you might want to round the probabilities to get class predictions
    class_predictions = (predictions >= 0.5).astype(int)
    print(title)
    print('AUC :', roc_auc_score(y, predictions))
    print('accuracy :', accuracy_score(y, class_predictions))
    print('precision :', precision_score(y, class_predictions, zero_division=0))
    print('recall :', recall_score(y, class_predictions))
    print('f1 :', f1_score(y, class_predictions))
    print('spec :', specificity_score(y, class_predictions))
    print('rmse :', rmse(y, predictions))

#defining a generic Function to give ROC_AUC Scores in table format for better readability
def crossvalscore(model, X, y, title):
    y = np.ravel(y)
    rmse = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1))
    scoring = make_scorer(precision_score, zero_division=0)
    specificity = make_scorer(specificity_score)
    scores = cross_val_score(model,X,y,cv=5,scoring='roc_auc',n_jobs=-1)
    acc = cross_val_score(model,X,y,cv=5,scoring='accuracy',n_jobs=-1)
    prec = cross_val_score(model,X,y,cv=5,scoring=scoring,n_jobs=-1)
    spec = cross_val_score(model,X,y,cv=5,scoring=specificity,n_jobs=-1)
    rec = cross_val_score(model, X, y, cv=5, scoring='recall', n_jobs=-1)
    f1 = cross_val_score(model, X, y, cv=5, scoring='f1', n_jobs=-1)
    rand_scores = pd.DataFrame({
    'cv':range(1,6),
    'roc_auc score':scores,
    'accuracy score':acc,
    'precision score':prec,
    'recall':rec,
    'f1': f1,
    'rmse': rmse,
    'spec' : spec
    })
    print(title)
    print('AUC :',rand_scores['roc_auc score'].mean())
    print('accuracy :',rand_scores['accuracy score'].mean())
    print('precision :', rand_scores['precision score'].mean())
    print('recall :', rand_scores['recall'].mean())
    print('f1 :', rand_scores['f1'].mean())
    print('spec :', rand_scores['spec'].mean())
    print('rmse :', rand_scores['rmse'].mean())
    print(rand_scores.sort_values(by='roc_auc score',ascending=False))

def objective(trial):
    x = trial.suggest_float("x", -7, 7)
    y = trial.suggest_float("y", -7, 7)
    return (x - 1) ** 2 + (y + 3) ** 2