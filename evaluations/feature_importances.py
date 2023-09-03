import numpy as np
import openpyxl
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance


def feature_importances(model, Xs, y, factors, title):
    perm_importance = permutation_importance(model.fit(Xs, y), Xs, y)

    features = np.array(factors)

    sorted_indices = perm_importance.importances_mean.argsort()[::-1]
    sorted_features = features[sorted_indices]
    sorted_importances = perm_importance.importances_mean[sorted_indices]

    # Print the importances with feature names
    for feature, importance in zip(sorted_features, sorted_importances):
        print(f"{feature}: {importance}")

    # Plot the feature importances in a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_features, sorted_importances, color='skyblue')
    plt.xlabel('Feature')
    plt.ylabel('Permutation Importance')
    plt.title('Feature Importances for Tuned Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    temp_file = f'${title}_feature_importances.png'
    plt.savefig(temp_file)
    plt.close()