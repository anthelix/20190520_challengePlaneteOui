"""
BCM custom metric file (weighted Mean Aboslute Error)
"""

import numpy as np


def weighted_mean_absolute_error(dataframe_1, dataframe_2):
    """Weighted mean absolute error regression loss

        ----------
        y_true : array-like of shape = (n_samples,2)
        Ground truth (correct) target values.
        y_pred : array-like of shape = (n_samples,2)
        Estimated target values.

        """

    y_true = dataframe_1.values
    y_pred = dataframe_2.values
    c12 = np.array([1136987, 1364719])

    return 2 * metrics.mean_absolute_error(y_true*c12, y_pred*c12) / np.sum(c12)




if __name__ == '__main__':
    import pandas as pd
    CSV_FILE_1 = '--------.csv'
    CSV_FILE_2 = '--------.csv'
    df_1 = pd.read_csv(CSV_FILE_1, index_col=0, sep=',')
    df_2 = pd.read_csv(CSV_FILE_2, index_col=0, sep=',')
    print(weighted_mean_absolute_error(df_1, df_2))
