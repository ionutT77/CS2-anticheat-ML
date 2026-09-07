import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix


def ranking(y,p):
    return {'roc_auc':float(roc_auc_score(y,p)), 'average_precision':float(average_precision_score(y,p)),
            'positive_prevalence':float(np.mean(y)), 'n':len(y), 'positives':int(np.sum(y))}


def threshold_metrics(y,p,threshold):
    tn,fp,fn,tp=confusion_matrix(y,p>=threshold,labels=[0,1]).ravel()
    return {'threshold':float(threshold),'accuracy':float((tp+tn)/len(y)),
            'precision':float(tp/max(tp+fp,1)),'recall':float(tp/max(tp+fn,1)),
            'fpr':float(fp/max(fp+tn,1)), 'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]],
            'precision_at_assumed_1pct_prevalence':float((.01*tp/max(tp+fn,1))/
                max(.01*tp/max(tp+fn,1)+.99*fp/max(fp+tn,1),1e-20))}


def exploratory_precision_threshold(y,p,target=.99):
    order=np.argsort(-p,kind='stable');yy=y[order];pp=p[order]
    end=np.r_[np.flatnonzero(np.diff(pp)!=0),len(pp)-1]
    tp=np.cumsum(yy)[end];precision=tp/(end+1)
    ok=(tp>0)&(precision>=target)
    return float(pp[end[np.flatnonzero(ok)[-1]]]) if ok.any() else float(np.nextafter(pp.max(),np.inf))


def weighted_auc(y,p):
    order=np.argsort(p,kind='stable');yy=y[order];pp=p[order]
    starts=np.r_[0,np.flatnonzero(np.diff(pp)!=0)+1]
    def calculate(weights):
        w=weights[order];pos=np.add.reduceat(w*yy,starts);neg=np.add.reduceat(w*(1-yy),starts)
        return np.sum(pos*(np.cumsum(neg)-neg/2))/(pos.sum()*neg.sum())
    return calculate


def cluster_bootstrap(y,probabilities,groups,iterations=2000):
    unique,inverse=np.unique(groups,return_inverse=True)
    funcs={name:weighted_auc(y,p) for name,p in probabilities.items()}
    rng=np.random.default_rng(9072026);samples={k:[] for k in funcs}
    differences={k:[] for k in funcs if k!='selected'}
    for _ in range(iterations):
        counts=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique));w=counts[inverse]
        if (w*y).sum()==0 or (w*(1-y)).sum()==0:continue
        values={k:fn(w) for k,fn in funcs.items()}
        for k,v in values.items():samples[k].append(v)
        for k in differences:differences[k].append(values['selected']-values[k])
    return {'auc_95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in samples.items()},
            'selected_minus_other_95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in differences.items()},
            'clusters':len(unique),'iterations':iterations}
