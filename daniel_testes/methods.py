from sklearn.metrics import r2_score, mean_absolute_error

def adjusted_r2(labels_test, labels_pred,data_train):

    adj_r2 = (1 - ((1 - r2_score(labels_test, labels_pred)) * (len(labels_test) - 1)) / 
            (len(labels_test) - data_train.shape[1] - 1))

    return adj_r2

#MAE = mean_absolute_error(labels_test,lebels_pred)