import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, explained_variance_score, precision_score, mean_absolute_error, classification_report, RocCurveDisplay, confusion_matrix,  recall_score, accuracy_score, roc_auc_score, precision_recall_curve, f1_score, mean_squared_error
import pandas as pd
import numpy as np

def plot_balancing(df, var):
    plt.figure()
    target_count = df[var].value_counts()
    target_count.plot(kind='bar', title=var);
    print(f"{len(df.loc[df[var] == 1]) / len(df)*100}%")

def plot_roc(y, hat_prob_y):
    RocCurveDisplay.from_predictions(
        y,
        hat_prob_y,
        name="ROC curve",
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.show()

def plot_confusion_matrix(hat_y, y, target_names):
    matrix = confusion_matrix(y, hat_y)
    sns.heatmap(matrix.T, square=True, annot=True, fmt="d", cbar=False,
    xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("true label")
    plt.ylabel("predicted label")

def estimate_cost(hat_y, y, cost_FP, cost_FN):
    return  np.sum(np.multiply(hat_y, (1 - y)) * cost_FP) + np.sum(np.multiply((1 - hat_y), y) * cost_FN)

def plot_to_evaluate(y, y_hat, title):
  print(f"{title} costs: {estimate_cost(y_hat, y, 1, 1)}")
  print(f"{title} MSE:{mean_squared_error(y, y_hat)}")
  print(f"{title} RMSE:{mean_squared_error(y, y_hat)**0.5}")
  print(f"report: {classification_report(y, y_hat)}")
  print(f"{title} accuracy score: {accuracy_score(y, y_hat)}")

def print_results(y, y_hat, title):
  print(f"{title} MAE:{mean_absolute_error(y, y_hat)}")
  print(f"{title} MSE:{mean_squared_error(y, y_hat)}")
  print(f"{title} RMSE:{mean_squared_error(y, y_hat)**0.5}")
  print(f"{title} Explained Variance Score:{explained_variance_score(y, y_hat)}")
  print(f"{title} R2: {r2_score(y, y_hat)}")

def plot_precision_recall_curve(probs, test_Y, label):
    plt.figure()
    precision, recall, _ = precision_recall_curve(test_Y, probs)
    no_skill = len(test_Y[test_Y == 1]) / len(test_Y)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

def plot_optimized_characteristics(tuned_delayed_model, X_val):
    best_params = tuned_delayed_model.best_params_
    importances = tuned_delayed_model.best_estimator_.feature_importances_
    print("Best parameters:")
    print(best_params)
    # summarize feature importance
    for i, v in enumerate(importances):
        print(f"{i} {X_val.columns[i]} Score: {v}")
    # plot feature importance
    plt.bar([x for x in range(len(importances))], importances)
    plt.show()
