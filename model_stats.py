import config

import os, sys, yaml, logging

from sklearn.metrics import (brier_score_loss, confusion_matrix, precision_score,
                             recall_score, f1_score, precision_recall_curve,
                             accuracy_score, roc_curve, roc_auc_score)
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import pprint

import numpy as np
import pandas as pd
import re

def show_scores(y_true, y_pred, y_prob):
    """Displays statistics on how a model performs against a
       particular dataset.

    Args:
      y_true: True labels
      y_pred: Predicted labels
      y_prob: Predicted probabilities

    """
    print("Accuracy: %s" % accuracy_score(y_true, y_pred))
    print("Precision: %s" % precision_score(y_true, y_pred))
    print("Recall: %s" % recall_score(y_true, y_pred))
    print("F1 Score: %s\n" % f1_score(y_true, y_pred))
    
    confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['True '], colnames=['Predicted-->'], margins=True)
    print(confusion_matrix)
    # Plot Evaluation
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
    auc = roc_auc_score(y_true,  y_prob[:, 1])
    fig, ax = plt.subplots(figsize = (10, 5))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax.grid()
    plt.legend(loc='best')
    plt.title('Receiver operating characteristic')
    plt.show()


def show_model_stats(clf, x_train, y_train, x_test, y_test):
    """Displays statistics for a model.

    Args:
      clf: Classifier
      x_train: Training features
      y_train: Training labels
      x_test: Test features
      y_test: Test labels

    """

    y_train_pred = clf.predict(x_train).flatten()
    y_test_pred = clf.predict(x_test).flatten()
    y_train_prob = clf.predict_proba(x_train).flatten()
    y_test_prob = clf.predict_proba(x_test).flatten()
    
    print("Model parameters")
  #  pprint.pprint(clf.get_params())
    
    print("Most important features")
  #  feature_imp = sorted(zip(clf.feature_importances_, list(x_train.columns)), reverse=True)
  #  pprint.pprint(feature_imp[:25])

    print("\nScores for Training Data")
    show_scores(y_train, y_train_pred, y_train_prob)

    print("\nScores for Test Data")
    show_scores(y_test, y_test_pred, y_test_prob)
    
#    print("\nModel Error Analysis")
#    show_model_error_analysis(clf,  x_train, y_train, x_test, y_test)

def hist2d(title, rec, x_col, y_col):
    x = rec[x_col]
    y = rec[y_col]
    
    gridx = np.linspace(min(x),max(x),11)
    gridy = np.linspace(min(y),max(y),11)

    H, xedges, yedges = np.histogram2d(x, y, bins=[gridx, gridy])    
    plt.figure()
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    myextent  =[xedges[0],xedges[-1],yedges[0],yedges[-1]]
    plt.imshow(H.T,origin='low',extent=myextent,interpolation='nearest',aspect='auto')
    plt.plot(x,y,'r.')
    plt.title(title)
    plt.colorbar()    
    plt.show()
    
def reverse_dummy(X, prefix):
    temp = X.filter(regex=(r'\b'+prefix))
    return temp.apply(lambda row: '+'.join([col for col, b in zip(temp.columns, row) if b]),axis=1)

def plot_error_class_comparison(X, Y, tp, fp, tn, fn):
    lgr = RandomizedLogisticRegression()
    lgr.fit(X,Y)
    important_cols = list()
    important_cats = list()

    for s,f in sorted(zip(map(lambda x: round(x, 4), lgr.scores_), X.columns), reverse=True):
        if (s>0.5) or (len(important_cols) + len(important_cats)) < 5:
            if (f != 0) and (tp[f].dtype == np.int8):
                res = re.match('(\w+)_\w+', f)
                if res: 
                    cat = res.groups()[0]
                    important_cats.append(cat)
            else:   
                important_cols.append(f)
                
    # Remove duplicates                    
    important_cats = list(set(important_cats))
            
    for cat in important_cats:
        tp_combined = reverse_dummy(tp, cat).value_counts()
        tn_combined = reverse_dummy(tn, cat).value_counts()
        fp_combined = reverse_dummy(fp, cat).value_counts()
        fn_combined = reverse_dummy(fn, cat).value_counts()
        pd.DataFrame([tp_combined,tn_combined,fp_combined,fn_combined], index=["True Positive","True Negative","False Positive","False Negative"]).T.plot(title=cat, kind='bar',  sharex="true", colormap="viridis")
        
    for col in important_cols:   # This is a dummy column for a categorical feature
        tp_combined = tp[col].value_counts()
        tn_combined = tn[col].value_counts()
        fp_combined = fp[col].value_counts()
        fn_combined = fn[col].value_counts()
        if (tp[col].dtype == np.float):
            pd.DataFrame([tp_combined,tn_combined,fp_combined,fn_combined], index=["True Positive","True Negative","False Positive","False Negative"]).T.plot(title=col,  sharex="true", colormap="viridis")
        else:
            pd.DataFrame([tp_combined,tn_combined,fp_combined,fn_combined], index=["True Positive","True Negative","False Positive","False Negative"]).T.plot(title=col, kind='bar',  sharex="true", colormap="viridis")
            
            
def show_model_error_analysis(clf, x_train, y_train, x_test, y_test):
    
    y_pred = clf.predict(x_test)
    
    # Get error types
    # tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()    
    tp = x_test[(y_test == 1) & (y_pred == 1)]
    fp = x_test[(y_test == 0) & (y_pred == 1)]
    tn = x_test[(y_test == 0) & (y_pred == 0)]
    fn = x_test[(y_test == 1) & (y_pred == 0)]      
    
    # tp vs fp    
    print("\nTrue Positives VS False Positives\n")
    X = tp.append(fp)
    Y = pd.DataFrame(0, index=tp.index, columns=['true_label']).append(pd.DataFrame(1, index=fp.index, columns=['true_label']))
    plot_error_class_comparison(X, Y, tp, fp, tn, fn)
    
    # tn vs fn    
    print("\nTrue Negatives VS False Negatives\n")
    X = tn.append(fn)
    Y = pd.DataFrame(0, index=tn.index, columns=['true_label']).append(pd.DataFrame(1, index=fn.index, columns=['true_label']))
    plot_error_class_comparison(X, Y, tp, fp, tn, fn)
    
    cat = 'Age Buckets VS Error Types'
    tp_combined = reverse_dummy(tp, cat).value_counts()
    tn_combined = reverse_dummy(tn, cat).value_counts()
    fp_combined = reverse_dummy(fp, cat).value_counts()
    fn_combined = reverse_dummy(fn, cat).value_counts()    
    pd.DataFrame([tp_combined,tn_combined,fp_combined,fn_combined], index=["True Positive","True Negative","False Positive","False Negative"]).T.plot(title=cat, figsize=(12, 8), kind='bar',  sharex="true", colormap="viridis")

    fp['last_time'] -= (fp['last_time'] > 0)*2000
    tp['last_time'] -= (tp['last_time'] > 0)*2000
    fn['last_time'] -= (fn['last_time'] > 0)*2000
    tn['last_time'] -= (tn['last_time'] > 0)*2000
    
    hist2d("False Positives", fp, 'feature_1', 'feature_2')

def main():
    """Loads a model from a directory and displays statistics about it.
    """

    options = config.get_config()

    # Load saved model
    logging.debug('Reading input %s' % options.input)
    clf = joblib.load(options.input)
    logging.debug('Finished reading %s' % options.input)

    show_model_stats(clf, x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()