import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef
THRESHOLD = 0.5

##xgb、xgbclassifier、lgb都是一样的，只有lgbClassifier不一样

def tpr_weight_funtion_self(y_predict,dtrain):
    y_true=dtrain.get_label()
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 'TPR_SELF',-(0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3)

def tpr_weight_funtion2(y_true,y_predict):
    # print('y_true',y_true)
    # print('y_predict',y_predict)
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return -(0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3)

def metric_scores_self(estimator, X, y_true):
    y_pred = estimator.predict_proba(X)[:, 1]#, ntree_limit=estimator.best_ntree_limit
    d = pd.DataFrame()
    d['prob'] = list(y_pred)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    # 要将series转为list否则会报错
    PosAll = list(pd.Series(y).value_counts())[1]
    NegAll = list(pd.Series(y).value_counts())[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer - 0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer - 0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer - 0.01).idxmin()]
    return -(0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3)




def metric_self(y_true, y_pred,sample_weight=None):
    # print('y_true',y_true)
    # print('y_predict',y_predict)
    d = pd.DataFrame()
    d['prob'] = list(y_pred)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    #要将series转为list否则会报错
    PosAll = list(pd.Series(y).value_counts())[1]
    NegAll = list(pd.Series(y).value_counts())[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return -(0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3)
def tpr_weight_funtion_lc(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 'TPR',-(0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3),False

def tpr_weight_funtion_lgb_cv(y_predict,train_data):
    y_true=train_data.get_label()
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 'TPR',-(0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3),False




def evalmcc_min(y_true,preds,eps=1e-15, normalize=True, sample_weight=None,
             labels=None):
    labels = y_true
    return  -matthews_corrcoef(labels, preds > THRESHOLD)

def log_loss_temp(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None,
             labels=None):






    # Clipping
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)



    return y_pred
