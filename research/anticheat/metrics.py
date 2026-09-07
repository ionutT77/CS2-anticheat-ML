import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score, brier_score_loss,
    log_loss, confusion_matrix, roc_curve)


def ranking(y,p):
    fpr,tpr,_=roc_curve(y,p)
    return {"roc_auc":float(roc_auc_score(y,p)), "average_precision":float(average_precision_score(y,p)),
            "partial_auc_1pct":float(roc_auc_score(y,p,max_fpr=.01)),
            "roc_recall_at_1pct_fpr":float(tpr[fpr<=.01].max())}


def evaluate(y,p,threshold=.5):
    pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {**ranking(y,p),"threshold":float(threshold), "accuracy":float(accuracy_score(y,pred)),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "precision":float(precision_score(y,pred,zero_division=0)),
            "recall":float(recall_score(y,pred,zero_division=0)),
            "f1":float(f1_score(y,pred,zero_division=0)),"fpr":float(fp/(fp+tn)),
            "brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,np.clip(p,1e-7,1-1e-7))),
            "confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]}
