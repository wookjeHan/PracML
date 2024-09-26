from __future__ import division
import warnings
import openml
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score

# These functions are from https://github.com/meeliskull/prg (Original code from the authors)
# Slightly modified because of deprecated numpy functions

def precision(tp, fn, fp, tn):
    with np.errstate(divide='ignore', invalid='ignore'):
        return tp/(tp + fp)


def recall(tp, fn, fp, tn):
    with np.errstate(divide='ignore', invalid='ignore'):
        return tp/(tp + fn)


def precision_gain(tp, fn, fp, tn):
    """Calculates Precision Gain from the contingency table

    This function calculates Precision Gain from the entries of the contingency
    table: number of true positives (TP), false negatives (FN), false positives
    (FP), and true negatives (TN). More information on Precision-Recall-Gain
    curves and how to cite this work is available at
    http://www.cs.bris.ac.uk/~flach/PRGcurves/.
    """
    n_pos = tp + fn
    n_neg = fp + tn
    with np.errstate(divide='ignore', invalid='ignore'):
        prec_gain = 1. - (n_pos/n_neg) * (fp/tp)
    if len(prec_gain.shape) > 0 and prec_gain.shape[0] > 1:
        prec_gain[tn + fn == 0] = 0
    elif tn + fn == 0:
        prec_gain = 0
    return prec_gain


def recall_gain(tp, fn, fp, tn):
    """Calculates Recall Gain from the contingency table

    This function calculates Recall Gain from the entries of the contingency
    table: number of true positives (TP), false negatives (FN), false positives
    (FP), and true negatives (TN). More information on Precision-Recall-Gain
    curves and how to cite this work is available at
    http://www.cs.bris.ac.uk/~flach/PRGcurves/.

    Args:
        tp (float) or ([float]): True Positives
        fn (float) or ([float]): False Negatives
        fp (float) or ([float]): False Positives
        tn (float) or ([float]): True Negatives
    Returns:
        (float) or ([float])
    """
    n_pos = tp + fn
    n_neg = fp + tn
    with np.errstate(divide='ignore', invalid='ignore'):
        rg = 1. - (n_pos/n_neg) * (fn/tp)
    if rg.shape[0] > 1:
        rg[tn + fn == 0] = 1
    elif tn + fn == 0:
        rg = 1
    return rg


def create_segments(labels, pos_scores, neg_scores):
    n = labels.shape[0]
    # reorder labels and pos_scores by decreasing pos_scores, using increasing neg_scores in breaking ties
    new_order = np.lexsort((neg_scores, -pos_scores))
    labels = labels[new_order]
    pos_scores = pos_scores[new_order]
    neg_scores = neg_scores[new_order]
    # create a table of segments
    segments = {'pos_score': np.zeros(n), 'neg_score': np.zeros(n),
                'pos_count': np.zeros(n), 'neg_count': np.zeros(n)}
    j = -1
    for i, label in enumerate(labels):
        if ((i == 0) or (pos_scores[i-1] != pos_scores[i])
                     or (neg_scores[i-1] != neg_scores[i])):
            j += 1
            segments['pos_score'][j] = pos_scores[i]
            segments['neg_score'][j] = neg_scores[i]
        if label == 0:
            segments['neg_count'][j] += 1
        else:
            segments['pos_count'][j] += 1
    segments['pos_score'] = segments['pos_score'][0:j+1]
    segments['neg_score'] = segments['neg_score'][0:j+1]
    segments['pos_count'] = segments['pos_count'][0:j+1]
    segments['neg_count'] = segments['neg_count'][0:j+1]
    return segments


def get_point(points, index):
    keys = points.keys()
    point = np.zeros(len(keys))
    key_indices = dict()
    for i, key in enumerate(keys):
        point[i] = points[key][index]
        key_indices[key] = i
    return [point, key_indices]


def insert_point(new_point, key_indices, points, precision_gain=0,
        recall_gain=0, is_crossing=0):
    for key in key_indices.keys():
        points[key] = np.insert(points[key], 0, new_point[key_indices[key]])
    points['precision_gain'][0] = precision_gain
    points['recall_gain'][0] = recall_gain
    points['is_crossing'][0] = is_crossing
    new_order = np.lexsort((-points['precision_gain'],points['recall_gain']))
    for key in points.keys():
        points[key] = points[key][new_order]
    return points


def _create_crossing_points(points, n_pos, n_neg):
    n = n_pos+n_neg
    points['is_crossing'] = np.zeros(points['pos_score'].shape[0])
    # introduce a crossing point at the crossing through the y-axis
    j = np.amin(np.where(points['recall_gain'] >= 0)[0])
    if points['recall_gain'][j] > 0:  # otherwise there is a point on the boundary and no need for a crossing point
        [point_1, key_indices_1] = get_point(points, j)
        [point_2, key_indices_2] = get_point(points, j-1)
        delta = point_1 - point_2
        if delta[key_indices_1['TP']] > 0:
            alpha = (n_pos*n_pos/n - points['TP'][j-1]) / delta[key_indices_1['TP']]
        else:
            alpha = 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_point = point_2 + alpha*delta

        new_prec_gain = precision_gain(new_point[key_indices_1['TP']], new_point[key_indices_1['FN']],
                                       new_point[key_indices_1['FP']], new_point[key_indices_1['TN']])
        points = insert_point(new_point, key_indices_1, points,
                              precision_gain=new_prec_gain, is_crossing=1)

    # now introduce crossing points at the crossings through the non-negative part of the x-axis
    x = points['recall_gain']
    y = points['precision_gain']
    temp_y_0 = np.append(y, 0)
    temp_0_y = np.append(0, y)
    temp_1_x = np.append(1, x)
    with np.errstate(invalid='ignore'):
        indices = np.where(np.logical_and((temp_y_0 * temp_0_y < 0), (temp_1_x >= 0)))[0]
    for i in indices:
        cross_x = x[i-1] + (-y[i-1]) / (y[i] - y[i-1]) * (x[i] - x[i-1])
        [point_1, key_indices_1] = get_point(points, i)
        [point_2, key_indices_2] = get_point(points, i-1)
        delta = point_1 - point_2
        if delta[key_indices_1['TP']] > 0:
            alpha = (n_pos * n_pos / (n - n_neg * cross_x) - points['TP'][i-1]) / delta[key_indices_1['TP']]
        else:
            alpha = (n_neg / n_pos * points['TP'][i-1] - points['FP'][i-1]) / delta[key_indices_1['FP']]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_point = point_2 + alpha*delta

        new_rec_gain = recall_gain(new_point[key_indices_1['TP']], new_point[key_indices_1['FN']],
                                   new_point[key_indices_1['FP']], new_point[key_indices_1['TN']])
        points = insert_point(new_point, key_indices_1, points,
                              recall_gain=new_rec_gain, is_crossing=1)
        i += 1
        indices += 1
        x = points['recall_gain']
        y = points['precision_gain']
        temp_y_0 = np.append(y, 0)
        temp_0_y = np.append(0, y)
        temp_1_x = np.append(1, x)
    return points


def create_prg_curve(labels, pos_scores, neg_scores=[]):
    """Precision-Recall-Gain curve

    This function creates the Precision-Recall-Gain curve from the vector of
    labels and vector of scores where higher score indicates a higher
    probability to be positive. More information on Precision-Recall-Gain
    curves and how to cite this work is available at
    http://www.cs.bris.ac.uk/~flach/PRGcurves/.
    """
    create_crossing_points = True # do it always because calc_auprg otherwise gives the wrong result
    if len(neg_scores) == 0:
        neg_scores = -pos_scores
    n = labels.shape[0]
    n_pos = np.sum(labels)
    n_neg = n - n_pos
    # convert negative labels into 0s
    labels = 1 * (labels == 1)
    segments = create_segments(labels, pos_scores, neg_scores)
    # calculate recall gains and precision gains for all thresholds
    points = dict()
    points['pos_score'] = np.insert(segments['pos_score'], 0, np.inf)
    points['neg_score'] = np.insert(segments['neg_score'], 0, -np.inf)
    points['TP'] = np.insert(np.cumsum(segments['pos_count']), 0, 0)
    points['FP'] = np.insert(np.cumsum(segments['neg_count']), 0, 0)
    points['FN'] = n_pos - points['TP']
    points['TN'] = n_neg - points['FP']
    points['precision'] = precision(points['TP'], points['FN'], points['FP'], points['TN'])
    points['recall'] = recall(points['TP'], points['FN'], points['FP'], points['TN'])
    points['precision_gain'] = precision_gain(points['TP'], points['FN'], points['FP'], points['TN'])
    points['recall_gain'] = recall_gain(points['TP'], points['FN'], points['FP'], points['TN'])
    if create_crossing_points == True:
        points = _create_crossing_points(points, n_pos, n_neg)
    else:
        points['pos_score'] = points['pos_score'][1:]
        points['neg_score'] = points['neg_score'][1:]
        points['TP'] = points['TP'][1:]
        points['FP'] = points['FP'][1:]
        points['FN'] = points['FN'][1:]
        points['TN'] = points['TN'][1:]
        points['precision_gain'] = points['precision_gain'][1:]
        points['recall_gain'] = points['recall_gain'][1:]
    with np.errstate(invalid='ignore'):
        points['in_unit_square'] = np.logical_and(points['recall_gain'] >= 0,
                                              points['precision_gain'] >= 0)
    return points


def calc_auprg(prg_curve):
    """Calculate area under the Precision-Recall-Gain curve

    This function calculates the area under the Precision-Recall-Gain curve
    from the results of the function create_prg_curve. More information on
    Precision-Recall-Gain curves and how to cite this work is available at
    http://www.cs.bris.ac.uk/~flach/PRGcurves/.
    """
    area = 0
    recall_gain = prg_curve['recall_gain']
    precision_gain = prg_curve['precision_gain']
    for i in range(1,len(recall_gain)):
        if (not np.isnan(recall_gain[i-1])) and (recall_gain[i-1]>=0):
            width = recall_gain[i]-recall_gain[i-1]
            height = (precision_gain[i]+precision_gain[i-1])/2
            area += width*height
    return(area)


# These codes are from me (Wookje Han, wh2571)
# Q2-2
dataset = openml.datasets.get_dataset("credit-g")
data, _, _, _ = dataset.get_data(dataset_format="dataframe")
# Let's shuffle, split it to X / Y
data = data.sample(frac=1)
Y = data['class']
X = data.drop('class', axis=1)
# Let's transform to numeric
Y = Y.map({'good':0, 'bad':1}).to_numpy()
# Also all for category columns
cat_features = X.select_dtypes(['category']).columns
X[cat_features] = X[cat_features].apply(lambda x: x.cat.codes)
X = X.to_numpy()
# Let's split to train / test (8, 2)
train_size = int(len(X)*8/10)
train_Xs = X[:train_size, :]
test_Xs = X[train_size:, :]
train_Ys = Y[:train_size]
test_Ys = Y[train_size:]
# Scaling features for stability
scalar = StandardScaler()
train_Xs = scalar.fit_transform(train_Xs)
test_Xs = scalar.transform(test_Xs)
# Now make models Adaboost, Logistic Regression
ada_clf = AdaBoostClassifier(algorithm="SAMME")
lr_clf = LogisticRegression()
# Let's train it
ada_clf = ada_clf.fit(train_Xs, train_Ys)
lr_clf = lr_clf.fit(train_Xs, train_Ys)
# Let's draw plot
ada_pred = ada_clf.predict_proba(test_Xs)
lr_pred = lr_clf.predict_proba(test_Xs)
# This is a classifier where only returns 1
apc = [1.0 for i in range(len(lr_pred))]
tn, fp, fn, tp = confusion_matrix(test_Ys, apc).ravel()
tpr_apc = tp/(tp+fn)
fpr_apc = fp/(fp+tn)
fpr_ada, tpr_ada, thresholds_ada = metrics.roc_curve(test_Ys, ada_pred[:, 1])
fpr_lr, tpr_lr, thresholds_lr = metrics.roc_curve(test_Ys, lr_pred[:,1])
plt.plot(fpr_ada, tpr_ada, label='Adaboost')
plt.plot(fpr_lr, tpr_lr, label='LinearRegression')
plt.scatter(fpr_apc, tpr_apc, label='all positive classifier', color='red')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.savefig("ROC CURVE.png")

plt.clf()
prec_ada, rec_ada, _ = precision_recall_curve(test_Ys, ada_pred[:, 1])
prec_lr, rec_lr, _ = precision_recall_curve(test_Ys, lr_pred[:, 1])
rec_apc = tp/(tp+fn)
prec_apc = tp/(tp+fp)
plt.plot(rec_ada, prec_ada, label='Adaboost')
plt.plot(rec_lr, prec_lr, label='LinearRegression')
plt.scatter(rec_apc, prec_apc, label='all positive classifier', color='red')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve")
plt.legend()
plt.savefig("PR CURVE.png")

# Q2-3
# Calculate AUROC
auc_ada = roc_auc_score(test_Ys, ada_pred[:, 1])
auc_lr = roc_auc_score(test_Ys, lr_pred[:, 1])
print("AUROC SCORE : ", auc_ada, auc_lr)
# Calculate PR
auc_ada = metrics.auc(rec_ada, prec_ada)
auc_lr = metrics.auc(rec_lr, prec_lr)
print("AUPR SCORE : ", auc_ada, auc_lr)
# Calculate AUPRG
prg_curve = create_prg_curve(test_Ys, ada_pred[:,1])
auprg_ada = calc_auprg(prg_curve)

prg_curve = create_prg_curve(test_Ys, lr_pred[:,1])
auprg_lr = calc_auprg(prg_curve)
print("AUPRG SCORE : ", auprg_ada, auprg_lr)