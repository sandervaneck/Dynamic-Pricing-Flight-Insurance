from imblearn.combine import SMOTETomek
from excelWriters.plotter import plot_balancing

def resample_combination(X, Y, workbook, file):
    # sheet = workbook.create_sheet(title='Table')
    # sheet.title = 'Refund balance'
    # plot_balancing(Y, workbook, 1, sheet, file)
    # Balance training set
    smt = SMOTETomek(random_state=10)
    X_train, y_train = smt.fit_resample(X, Y)
    return X_train, y_train
    # plot_balancing(y_train, workbook, 26, sheet, file)
