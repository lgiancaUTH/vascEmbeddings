from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, roc_auc_score, cohen_kappa_score
from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt

def sigTestAUC(data1, data2, disp='long'):
    '''
    return a string with AUC and significance based on the Mann Whitney test
    disp= short|long|auc
    '''
    u, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    # p_value *= 2 # no longer required

    p_val_str = ''
    pValStars = ''
    if (p_value <= 0.001):
        p_val_str = '***p<0.001'
        pValStars = '***'
    elif (p_value <= 0.01):
        p_val_str = '**p<0.01'
        pValStars = '**'
    elif (p_value <= 0.05):
        p_val_str = '*p<0.05'
        pValStars = '*'
    else:
        p_val_str = 'not sig. p={:0.2f}'.format(p_value)
        pValStars = ''

    aucVal = 1 - u / (len(data1) * len(data2))

    if disp == 'short':
        strOut = '{:0.2f}{:}'.format(aucVal, pValStars)
    elif disp == 'long':
        strOut = '{:0.2f} ({:})'.format(aucVal, p_val_str)
    else:
        strOut = '{:0.2f}'.format(aucVal)

    return strOut


def findCutoffPnt3(dataPos, dataNeg):
    """
    Find cutoff point minimizing the distance to Sens 1, spec 1 and calculate statistics (with kappa, code based on findCutoffPnt2).
    format confMat:
     array([[TN, FP],
            [ FN, TP]]))
    :param dataPos:
    :param dataNeg:
    :param dataPosGtNeg:
    :return: acc,sens,spec,roc_auc, cutoffTh, confusionMat, kappa
    """

    dataAll = np.concatenate((dataPos, dataNeg))
    lblArr = np.zeros(len(dataAll), dtype=bool)
    lblArr[0:len(dataPos)] = True

    fpr, tpr, thresholds = roc_curve(lblArr, dataAll, pos_label=True)
    roc_auc = auc(fpr, tpr)

    # invert comparison if (ROC<0.5) required
    if roc_auc < 0.5:
        lblArr = ~lblArr
        fpr, tpr, thresholds = roc_curve(lblArr, dataAll, pos_label=True)
        roc_auc = auc(fpr, tpr)
        print 'inverting labels'

    # calculate best cut-off based on distance to top corner of ROC curve
    distArr = np.sqrt(np.power(fpr, 2) + np.power((1 - tpr), 2))
    cutoffIdx = np.argsort(distArr)[0]
    cutoffTh = thresholds[cutoffIdx]

    lblOut = dataAll >= cutoffTh

    acc = accuracy_score(lblArr, lblOut)
    sens = tpr[cutoffIdx]
    spec = 1 - fpr[cutoffIdx]
    cfMat = confusion_matrix(lblArr, lblOut)

    kappa = cohen_kappa_score(lblOut, lblArr)

    return (acc, sens, spec, roc_auc, cutoffTh, cfMat, kappa)


def rocBootstrap(cntArr, pdArr, bootstrapsNumIn=1000):
    """
    ROC bootstrap
    :param cntArr:
    :param pdArr:
    :param bootstrapsNumIn:
    :return:
    """
    n_bootstraps = bootstrapsNumIn
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    bootRocLst = []
    y_true = np.append([0] * len(cntArr), [1] * len(pdArr))
    y_pred = np.append(cntArr, pdArr)
    rng = np.random.RandomState(rng_seed)
    # create fpr grid for interpolation
    fprGridVec = np.linspace(0, 1, 100, endpoint=True)
    # matrix containing all tpr corresponding to fprGridVec
    tprGridMat = np.zeros((len(fprGridVec), n_bootstraps))
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        tmpFpr, tmpTpr, _ = roc_curve(y_true[indices], y_pred[indices])
        tmpFpr = np.concatenate(([0], tmpFpr, [1]))
        tmpTpr = np.concatenate(([0], tmpTpr, [1]))

        # interpolate for comparable ROCs
        fInter = interp1d(tmpFpr, tmpTpr, kind='nearest')
        tprGridMat[:, i] = fInter(fprGridVec)

        bootstrapped_scores.append(score)
        bootRocLst.append([tmpFpr, tmpTpr])
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    # confidence interval for AUCs
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]

    return (confidence_lower, confidence_upper, fprGridVec, tprGridMat)


def plotRocAndConf(fprGridVec, tprGridMat, labelIn=''):
    """
    Plot ROC curve with confidence intervals. Requires the ROC boostrapped with rocBootstrap
    :param fprGridVec:
    :param tprGridMat:
    :param labelIn:
    :return:
    """
    n_bootstraps = tprGridMat.shape[1]

    # confidence interval for ROC
    tprGridMatS = np.sort(tprGridMat, axis=1)
    tprLow05 = tprGridMatS[:, int(0.05 * n_bootstraps)]
    tprTop95 = tprGridMatS[:, int(0.95 * n_bootstraps)]
    tprMean = np.mean(tprGridMat, axis=1)

    plt.hold(True)
    ax = plt.gca()  # kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(fprGridVec, tprMean, '-', linewidth=4, label=labelIn)
    ax.fill_between(fprGridVec, tprLow05, tprTop95, facecolor=base_line.get_color(), alpha=0.2)