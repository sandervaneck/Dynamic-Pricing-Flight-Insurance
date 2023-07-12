from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import numpy as np

def model_rfc(train_X, train_Y):
    rfc = RandomForestClassifier()
    return rfc.fit(train_X, train_Y)

def cost_fun(y, hat_y):
    cost_FP = 100
    cost_FN = 100
    return np.sum(np.multiply(hat_y, (1 - y)) * cost_FP) + np.sum(np.multiply((1 - hat_y), y) * cost_FN)


def optimized_random_forest(X_train, Y_train):

    rfc = RandomForestClassifier()
    learning_rates = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
    max_depths = [3, 4, 5, 6, 7, 8]
    min_samples_splits = [3, 4, 5, 6, 7, 8]
    min_samples_leafs = [1, 2, 3, 4, 5, 6, 7, 8]
    criterions = ['gini', 'entropy']

    param_grid = {
        "criterion": criterions,
        "max_depth": max_depths,
        "min_samples_split": min_samples_splits,
        "min_samples_leaf": min_samples_leafs,
    }
    basemodel = rfc
    tuned_delayed_model = HalvingGridSearchCV(basemodel, param_grid,
                                              scoring=make_scorer(cost_fun, greater_is_better=True), n_jobs=-1,
                                              min_resources="smallest", factor=9, cv=6)

    return tuned_delayed_model.fit(X_train, Y_train)