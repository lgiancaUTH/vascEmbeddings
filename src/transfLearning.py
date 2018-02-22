###################################################
#
#   Script to
#   - CTrain test the transfer learning algorithm vessel -> Diagnosis
#
#
#
##################################################

#Python
import pandas as pd
import numpy as np
import sys
import h5py
import copy
import os
import matplotlib.pyplot as plt


# SKlearn
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.covariance import GraphLassoCV
import sklearn.linear_model as lm
import sklearn.metrics as met
import sklearn.manifold as man
import sklearn.svm as svm
import sklearn.ensemble as ens
import sklearn.preprocessing as pre
import sklearn.decomposition as dec
import sklearn.neighbors as nei

# Keras
import keras as k
import tensorflow as tf
assert k.backend.image_dim_ordering() == 'tf'
# (rows, cols, channels)
# (conv_dim1, conv_dim2, conv_dim3, channels)
import keras.models as km
import keras.layers as kl
import keras.layers.merge as klm
import keras.backend as K
import keras.regularizers as regularizers
import keras.callbacks as callbk
import keras.preprocessing as kp

import skimage.io as skio
import skimage.transform as sktr
from skimage.filters import  threshold_otsu

# visualization
import seaborn as sns
sns.set_context("notebook", font_scale=1.5  )
# sns.set_style("whitegrid")


# custom
import segmentVasculature as sv
import utilsStats
import utils




sys.path.insert(0, './lib/')


class Messidor:
    """
    Messidor dataset
    """
    def __init__(self):
        # File location
        self.baseDir = '/data/bigdata/retina/messidor/'
        self.gtFile = self.baseDir + 'Annotation_Full.csv'
        self.imgDir = self.baseDir + 'imgs/'

        # load GT (and rename columns)
        self.gtFr = pd.read_csv( self.gtFile )
        self.gtFr.columns = pd.Index(['img', 'department', 'retinopathy', 'edemaRisk'])
        pass

    def loadInfo(self, imgID):
        """
        Load information
        :param imgID: numerical id
        :return: information about image as dictionary
        """
        return self.gtFr.loc[imgID].to_dict()

    def loadImg(self, imgID):
        """
        load image
        :param imgID: numerical id
        :return: (info, image)
        """
        info = self.loadInfo(imgID)

        return skio.imread( self.imgDir + info['img'])



class MessidorDR(Messidor):
    """
    Messidor dataset with information about DR from previous algorithms
    """
    def __init__(self):
        Messidor.__init__(self)
        self.drFileInfo = './data/drDiagnosisMessidor.csv'

        # load
        drDiagFr = pd.read_csv(self.drFileInfo)
        # join
        self.gtFr = self.gtFr.join(drDiagFr.set_index('img'), on='img' )



def upperTrianFeatVec( connArr ):
    """
    Convert symmetrical square Mat to feat vector
    """
    lblNum = connArr.shape[0]
    # find upper indices
    indConn = np.triu_indices( lblNum, 0 )
    indConnNum = len(indConn[0])
    # linearize
    X = connArr[indConn].tolist()
    
    return X

def generateEncoding( mesIn, savedEncFile=None, flipImgArr=None ):
    """
    Generate encoding and feature matrix
    :param mesIn: Messidor object
    :param savedEncFile: file with the preencoded vessels (set to none not to use it)
    :return: feature matrix
    """

    modelEnc = sv.genEncModel()
    if modelEnc is  None:
        print 'encoding layer not available'
        return None

    # get gt
    mesFr = mesIn.gtFr

    # feat Matrix init
    X = []

    # check if encoding available
    hf = None
    if (savedEncFile is not None) and (os.path.exists( savedEncFile )):
        # start reading/writing
        hf = h5py.File(savedEncFile, "r+")
    elif savedEncFile is not None:
        # start writing
        hf = h5py.File(savedEncFile, "w")

    # generate encoding and feat Matrix
    for imgID in mesFr.index.values:
        print 'loading ', imgID
        imgIDstr = str(imgID)

        # calculate/load encoding
        resEnc = None
        if  (hf is not None) and imgIDstr in hf:
            resEnc = hf[imgIDstr][:]
        else:
            img = mesIn.loadImg( imgID )
            # flip image
            if (flipImgArr is not None) and flipImgArr[imgID]:
                img = np.fliplr(img)
                print 'flipping ', imgID
            
            resEnc, coordEnc = sv.getImageEncoding(img, modelEnc)

        # save encoding
        if (hf is not None) and (imgIDstr not in hf):
            hf.create_dataset(imgIDstr, data=resEnc)
            print 'saving encoding ',  imgIDstr


        #- filter out "empty vectors"
        coeffSum = np.sum(np.abs(resEnc), axis=1) # sum across vectors
        th = threshold_otsu(coeffSum)
        resEnc = resEnc[coeffSum < th, :]
        #-



        #- Statistics as features
        q75, q50, q25, q1, q99 = np.percentile(resEnc, [75, 50, 25, 1, 99], axis=0, keepdims=False)
        iqr = q75-q25
        # feature vector
        statFeatVec = np.append(q50, iqr)

        # statFeatVec = np.append(q1, q99)
        #-
        
        #- Covariance as features
        covMat = np.cov(resEnc.T)
        covFeatVec = upperTrianFeatVec(covMat)
        #-
        
        # current feat vec
        featVec = statFeatVec
        # featVec = covFeatVec
        
        # grow feat matrix
        if len(X) == 0:
            X = featVec
        else:
            X = np.vstack( (X,featVec) )

    # close hf if needed
    if hf is not None:
        hf.close()

    return X

def saveH5( varIn, varName, fileOut ):
    """
    Save variable as H5 (HDF)
    :param varIn: variable
    :param varName: "name-of-dataset"
    :param fileOut: 'name-of-file.h5'
    :return:
    """

    with h5py.File(fileOut, 'w') as hf:
        hf.create_dataset(varName, data=varIn)

def loadH5( varName, fileIn ):
    """
    Save variable as H5 (HDF)
    :param varName: "name-of-dataset"
    :param fileIn: 'name-of-file.h5'
    :return: variable
    """

    with h5py.File(fileIn, 'r') as hf:
        data = hf[varName][:]

    return data


def crossValidation( X, y ):
    """
    GCross validate
    :param mesIn: Messidor object
    :param featMatIn: feature matrix corresponding to the indices in mesIn.
    :return:
    """

    # Param
    N_SPLITS = 50
    RND_SEED = 6543215468
    #


    # --- Validation
    skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RND_SEED)

    mdlDic = {}
    # liblinear coordinate descent
    mdlDic[0] = {'name': 'Logistic Regression (L1 reg.)', 'auc': [], 'scores': [], 'featWeight': [], 'rndFeatWeights': [],
                 'scoresOut': [], \
                 'model': lm.LogisticRegression(penalty='l1'), 'y': []}
    # liblinear coordinate descent
    mdlDic[1] = {'name': 'Logistic Regression (L2 reg.)', 'auc': [], 'scores': [], 'featWeight': [], 'rndFeatWeights': [],
                 'scoresOut': [], \
                 'model': lm.LogisticRegression(penalty='l2'), 'y': []}
    # # liblinear coordinate descent
    # mdlDic[2] = {'name': 'Elastic Net', 'auc': [], 'scores': [], 'featWeight': [], 'rndFeatWeights': [], 'scoresOut': [],\
    # #              'model': lm.ElasticNet(alpha=1, l1_ratio=0.0001, fit_intercept=True, normalize=True), 'y': []}
    #             'model': lm.ElasticNet(alpha=1, l1_ratio=0.005, fit_intercept=True, normalize=False), 'y': []}

    mdlDic[3] = {'name': 'Linear SVM', 'auc': [], 'scores': [], 'featWeight': [], 'rndFeatWeights': [], 'scoresOut': [], \
                 'model': svm.SVC(kernel="linear", probability=True), 'y': []}

    mdlDic[4] = {'name': 'Random Forest Classifier', 'auc': [], 'scores': [], 'featWeight': [], 'rndFeatWeights': [],
                 'scoresOut': [], \
                 'model': ens.RandomForestClassifier(), 'y': []}
    # gradient descent
    mdlDic[2] = {'name': 'Logistic Regression (Elastic Net)', 'auc': [], 'scores': [], 'featWeight': [],
                 'rndFeatWeights': [], 'scoresOut': [], \
                 'model': lm.SGDClassifier(loss='log', penalty='elasticnet', warm_start=False),
                 'y': []}

    # mdlDic[6] = {'name': 'SVC RBF', 'auc': [], 'scores': [], 'featWeight': [], 'rndFeatWeights': [], 'scoresOut': [], \
    #              'model': svm.SVC(kernel="rbf", C=0.0001, probability=True), 'y': []}

    # sdaMld = sda.SDA()
    # mdlDic[5] = {'name': 'SDA', 'auc': [], 'scores': [], 'featWeight': [], 'scoresOut': [], \
    #              'model': sdaMld}

    # estimate random weights if > 0. Number represent the iterations per fold
    NUM_RAND_WEIGHTS = 50

    yCV = []  # cross validated labels
    for train_index, test_index in skf.split(range(len(y)), y):
        # split
        trainX = X[train_index, :]
        trainY = y[train_index]
        testX = X[test_index, :]
        testY = y[test_index]

        # standardize
        #     scaler = pre.StandardScaler( with_mean=True, with_std=True )

        scaler = pre.RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
        scaler.fit(trainX)
        trainX = scaler.transform(trainX)
        testX = scaler.transform(testX)


        # # PCA transforming
        # pca = dec.PCA(n_components=5) # explain 99% of variance in the dataset
        # pca.fit(trainX)
        # trainX = pca.transform(trainX)
        # testX = pca.transform(testX)


        # store cross validate labels
        yCV = np.append(yCV, y[test_index])
        for mId in mdlDic:
            mdlDic[mId]['model'].fit(trainX, trainY)
            # detect probabilities (using predict_proba when available)
            p = None
            pOut = None
            if 'predict_proba' in dir(mdlDic[mId]['model']):
                p = mdlDic[mId]['model'].predict_proba(testX)[:, 1]
            else:
                p = mdlDic[mId]['model'].predict(testX)

            # # store coefficients (if available)
            # if 'coef_' in dir(mdlDic[mId]['model']):
            #     mdlDic[mId]['featWeight'].append(mdlDic[mId]['model'].coef_.flatten())


            mdlDic[mId]['scores'] = np.append(mdlDic[mId]['scores'], p)
            mdlDic[mId]['y'] = np.append(mdlDic[mId]['y'], y[test_index])
            # stack on 0-axis since they are the prediciotn on the same samples
            if mdlDic[mId]['scoresOut'] == []:
                mdlDic[mId]['scoresOut'] = pOut
            else:
                mdlDic[mId]['scoresOut'] = np.vstack((mdlDic[mId]['scoresOut'], pOut))

                # convert labels to numpy
    yCvArr = np.array(yCV)
    for mId in mdlDic:
        print '-' * 10, mdlDic[mId]['name']

        # global AUC/significance
        scoresArr = mdlDic[mId]['scores']
        # stats
        aucStr = utilsStats.sigTestAUC(scoresArr[yCvArr == 0], scoresArr[yCvArr == 1], disp='long')
        (acc, sens, spec, roc_auc, cutoffTh, cfMat, kappa) = utilsStats.findCutoffPnt3(scoresArr[yCvArr == 1],
                                                                                       scoresArr[yCvArr == 0]) \
            # store aucStr
        mdlDic[mId]['aucStr'] = aucStr

        print aucStr
        print 'roc_auc_t: {:0.3f}, sens: {:0.3f}, spec: {:0.3f}, cutoffTh: {:0.3f}, kappa: {:0.3f}, acc: {:0.3f}'.format(
            roc_auc, sens, spec, cutoffTh, kappa, acc)

    # ---

def tmpPlotInteractiveProj(mesIn, X, isInteractive=True):
    """
    Project features to 2D, display and allow for showing the image that generated the graph
    :param mesIn:
    :param X:
    :param y:
    :return:
    """
    N_NEIGH = 5 # number of neighbors to show
    HEIGHT_VIS = 700

    # pMod = dec.PCA( n_components=2 )
    # pMod = dec.KernelPCA(n_components=2, kernel='rbf')
    pMod = man.TSNE(n_components=2, random_state=0)

    X = pre.StandardScaler().fit_transform( X )
    Xp = pMod.fit_transform(X)
    # absolute path
    imgDir  = mesIn.imgDir
    # GT
    gtFr = mesIn.gtFr
    # fit nearest neighbor model to TSNE space
    nbrsMod = nei.NearestNeighbors(n_neighbors=N_NEIGH, algorithm='ball_tree').fit(Xp)

    def fun(idxLst):
        coordMeanVec = np.mean( Xp[idxLst,:], axis=0 )
        distVec, indDstVec = nbrsMod.kneighbors([coordMeanVec])

        selFr = gtFr.iloc[indDstVec.flatten()]
        print selFr['img']
        imgSelLst = []
        for (_, row), fIdx in zip(selFr.iterrows(), range(len(gtFr))):
            imgTmp  = skio.imread( imgDir + row['img'] )
            imgTmp = sktr.resize(imgTmp, (HEIGHT_VIS, int(HEIGHT_VIS*1.0/imgTmp.shape[0]*imgTmp.shape[1]))   )
            imgSelLst.append( imgTmp )
        plt.figure()
        skio.imshow( np.hstack( imgSelLst ) )
        plt.axis('off')

    #- generate labels
    y = np.zeros(len(gtFr), dtype=int)
    y[(gtFr['retinopathy'].values > 0) & (gtFr['edemaRisk'].values == 0)] = 1
    y[(gtFr['retinopathy'].values > 0) & (gtFr['edemaRisk'].values > 0)] = 2
    y[(gtFr['retinopathy'].values > 2)] = 3
    lblLst = ('No DR or ME', 'mild DR', 'DR and at risk of ME', 'High DR')
    # -

    if isInteractive:
        # interactive cannot show legend
        utils.plotScatterPicker( fun, x=Xp[:, 0], y=Xp[:, 1], c=y, picker=5, label=y)
        plt.legend()
    else:
        currPalette = sns.color_palette("binary", n_colors=len(lblLst))
        fig = plt.figure()
        # plt.hold(True)
        ax = fig.add_subplot(111)
        for idCol in np.unique(y):
            ax.scatter(Xp[y==idCol, 0], y=Xp[y==idCol, 1], c=currPalette[idCol], label=lblLst[idCol])
        plt.legend()

def findSimilarImgs( mesIn, X, nNeighIn=5, maxImgNum2Disp=10 ):
    """
    Find sinilar images irrespective of the 'left right' receive Messidor as inpur (rather than gtFr in findSimilarImgs)
    :param mesIn:
    :param X:
    :param nNeighIn:
    :param maxImgNum2Disp: maximum nuber of images to be displayed
    :return:
    """
    HEIGHT_VIS = 700
    N_NEIGH = nNeighIn # neighbour comparison

    # pMod = man.TSNE(n_components=2, random_state=0)
    # X = pre.StandardScaler().fit_transform( X )
    # Xp = pMod.fit_transform(X)
    Xp = X

    # absolute path
    imgDir  = mesIn.imgDir
    # GT
    gtFr = mesIn.gtFr


    # fit nearest neighbor model to TSNE space
    nbrsMod = nei.NearestNeighbors(n_neighbors=N_NEIGH, algorithm='ball_tree').fit(Xp)
    distVec, indDstVec = nbrsMod.kneighbors(Xp)

    closestNeighIndDst = np.argsort( np.sum(distVec[:, 1:N_NEIGH+1], axis=1) )
    closestNeighPairs = indDstVec[closestNeighIndDst,:]

    fileShownArr = np.array([])
    for i in range( maxImgNum2Disp ):

        fileLst = gtFr.iloc[closestNeighPairs[i, :]]['img'].values
        # check if images have been shown
        if len(np.intersect1d( fileShownArr, fileLst )):
            continue # skip
        # image names
        print i, fileLst

        imgSelLst = []
        for fTmp in fileLst:
            imgTmp  = skio.imread( imgDir+fTmp )
            imgTmp = sktr.resize(imgTmp, (HEIGHT_VIS, int(HEIGHT_VIS*1.0/imgTmp.shape[0]*imgTmp.shape[1]))   )
            imgSelLst.append( imgTmp )
        plt.figure()
        skio.imshow( np.hstack( imgSelLst ) )
        plt.axis('off')
        plt.title(str(fileLst))
        plt.show()
        # add seen images
        fileShownArr = np.append(fileShownArr, fileLst)


def rnnTrain( mesFr, y, savedEncFile, modelFile ):
    NAME_EXP = modelFile

    MAX_SMP_SIZE = 9400 # maximum samples per sequence
    BATCH_SIZE = 60
    FEAT_VEC_SIZE = 128

    #--- Model
    inputSeq = kl.Input(batch_shape=(BATCH_SIZE, MAX_SMP_SIZE, FEAT_VEC_SIZE), name='inputSeq')

    resp = kl.AveragePooling1D(pool_size=8, strides=None, padding='valid')( inputSeq ) # 8-fold temporal reduction
    resp = kl.Conv1D(filters=128, kernel_size=5, padding='valid', activation='relu',  strides=10)( resp )
    resp = kl.Dropout(rate=0.2)(resp)
    resp = kl.MaxPooling1D(pool_size=2, strides=None, padding='valid')( resp )
    resp = kl.Conv1D(filters=128, kernel_size=5, padding='valid', activation='relu', strides=10)(resp)
    resp = kl.Dropout(rate=0.2)(resp)
    resp = kl.MaxPooling1D(pool_size=2, strides=None, padding='valid')(resp)
    resp = kl.Flatten()(resp)
    resp = kl.Dense(32, activation='sigmoid')(resp)
    resp = kl.Dropout(rate=0.2)(resp)

    # resp = kl.LSTM(128, stateful = False, return_sequences = False)(resp)

    # resp = kl.Reshape((MAX_SMP_SIZE, FEAT_VEC_SIZE))(resp)
    # resp = kl.GRU(1024, stateful = False, return_sequences = True)(resp)
    # # model.add(kl.Dropout(0.2))
    # resp = kl.GRU(128, return_sequences = True, stateful=False)(resp)
    # resp = kl.GRU(128, stateful=False)(resp)
    respOut = kl.Dense(1, activation='sigmoid')(resp)
    model = km.Model(inputs=inputSeq, outputs=respOut)

    # save architecture to disk
    json_string = model.to_json()
    open('./data/' + NAME_EXP + '_architecture.json', 'w').write(json_string)
    #---

    from keras.utils import plot_model
    plot_model(model, to_file='modelRnn.png', show_shapes=True)
    print 'plot'

    hf = h5py.File(savedEncFile, "r")

    X = np.zeros((len(mesFr), MAX_SMP_SIZE, FEAT_VEC_SIZE) ) # feat matrix
    # generate encoding and feat Matrix
    for nId, imgID in enumerate(mesFr.index.values): # nId, incremental id
        print 'loading ', imgID
        imgIDstr = str(imgID)

        # load encoding
        resEnc = hf[imgIDstr][:]

        # - filter out "empty vectors"
        coeffSum = np.sum(np.abs(resEnc), axis=1)  # sum across vectors
        th = threshold_otsu(coeffSum)
        resEnc = resEnc[coeffSum < th, :]
        # -

        # add to feature matrix
        X[nId,:resEnc.shape[0],:] = resEnc

    # close hf if needed
    if hf is not None:
        hf.close()

    # adam = k.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #     cbEStop = callbk.EarlyStopping( monitor='val_loss', min_delta=0.001, patience=100 )
    cbEStop = callbk.EarlyStopping(monitor='loss', min_delta=0.01, patience=5)
    cbTB = callbk.TensorBoard(log_dir='./Graph', histogram_freq=1,write_graph=True)
    cbCheckpoint = callbk.ModelCheckpoint(filepath='./data/' + NAME_EXP + '_best_weights.h5',
                                   verbose=1, monitor='val_loss', mode='auto',
                                   save_best_only=True)  # save at each epoch if the validation decreased

    
    histModel = model.fit(X, y, epochs=150, validation_split=0.1, callbacks=[cbEStop,cbTB,cbCheckpoint], batch_size=BATCH_SIZE, verbose=True)  # original
    # p = model.predict_on_batch(testX)
    #
    # p = p[:, 1]
    # aucScore = met.roc_auc_score(testY, p)
    model.save_weights('./data/' + NAME_EXP + '_last_weights.h5', overwrite=True)

    return histModel, model



def rnnTest( mesFr, y, savedEncFile, modelIn ):
    MAX_SMP_SIZE = 9400 # maximum samples per sequence
    BATCH_SIZE = 60
    FEAT_VEC_SIZE = 128


    hf = h5py.File(savedEncFile, "r")
    X = np.zeros((len(mesFr), MAX_SMP_SIZE, FEAT_VEC_SIZE) ) # feat matrix
    # generate encoding and feat Matrix
    for nId, imgID in enumerate(mesFr.index.values): # nId, incremental id
        print 'loading ', imgID
        imgIDstr = str(imgID)

        # load encoding
        resEnc = hf[imgIDstr][:]

        # - filter out "empty vectors"
        coeffSum = np.sum(np.abs(resEnc), axis=1)  # sum across vectors
        th = threshold_otsu(coeffSum)
        resEnc = resEnc[coeffSum < th, :]
        # -

        # add to feature matrix
        X[nId,:resEnc.shape[0],:] = resEnc

    # close hf if needed
    if hf is not None:
        hf.close()

    p = modelIn.predict(X, batch_size=60)

    print met.roc_auc_score(y, p.flatten())

    return p

def crossValidationNN(mesFrIn, y, encFileIn, nameExp ='cnn1'):
    TRAIN_INFO_FILE =  './data/trainInfoFr.csv'
    # get train/test subset
    trainInfoFr = None
    if os.path.exists(TRAIN_INFO_FILE):
        trainInfoFr = pd.read_csv(TRAIN_INFO_FILE)
    else:
        trainInfoFr = mesFrIn.copy()
        trainInfoFr['train'] = 0
        randInd = np.random.choice(range(len(mes.gtFr)), 600, replace=False )
        trainInfoFr.set_value(randInd, 'train', 1)
        trainInfoFr.to_csv(TRAIN_INFO_FILE)

    mesTrainFr = mesFrIn.loc[trainInfoFr.train == 1]
    yTrain = y[trainInfoFr.train == 1]

    #-- train/load
    fileModel = './data/' + nameExp + '_architecture.json'
    fileWeights = './data/' + nameExp + '_best_weights.h5' # _best_weights.h5 or _last_weights.h5
    model = None
    if not os.path.exists(fileModel):
        histModel, model = rnnTrain(mesTrainFr, yTrain, encFileIn, nameExp)
    else:
        model = km.model_from_json(open(fileModel).read())
        model.load_weights(fileWeights)
    #--

    #-- test
    mesTestFr = mesFrIn.loc[trainInfoFr.train == 0]
    yTrain = y[trainInfoFr.train == 0]

    # to be fixed
    # yTrain = yTrain[0:720]
    # mesTestFr = mesTestFr[0:720]
    p = rnnTest(mesTestFr, yTrain, encFileIn, model)
    #--

    return p
if __name__ == '__main__':
    ENC_FILE = 'data/encVess.h5'
    # FEAT_MAT_FILE = 'data/featMatSmStatIqr99.h5'
    # FEAT_MAT_FILE = 'data/featMatSmStatMed.h5'
    # FEAT_MAT_FILE = 'data/featMatSmStatIqr.h5'
    FEAT_MAT_FILE = 'data/featMatSmStat.h5'
    # FEAT_MAT_FILE = 'data/featMatStat.h5'
    # FEAT_MAT_FILE = 'data/featMatCov.h5'


    # mes = Messidor()
    mes = MessidorDR()

    # Run cross validation
    print 'retinopathy 0 vs all'
    # set test
    y = (mes.gtFr['retinopathy']>0).values
    #y = (mes.gtFr['edemaRisk']>0).values


    # p = crossValidationNN(mes.gtFr, y, ENC_FILE, nameExp='cnn1')


    # generate encoding vectors
    X = []
    if not os.path.exists(FEAT_MAT_FILE):
        X = generateEncoding(mes, ENC_FILE)
        saveH5(X, 'featMat', FEAT_MAT_FILE)
    else:
        print 'loading ', FEAT_MAT_FILE
        X = loadH5('featMat', FEAT_MAT_FILE)

    # # add DR features
    # X = np.hstack((X, mes.gtFr[['drF0', 'drF1', 'drF2']].values))
    # # use only dr features
    # X = mes.gtFr[['drF0', 'drF1', 'drF2']].values


    # tmpPlotInteractiveProj(mes, X, isInteractive=False)
    # findSimilarImgs(mes, X, 5) # paper

    # print 'n=',len(y)
    # crossValidation(X, y)
    #
    print 'retinopathy 0 vs 3'
    lbl = (mes.gtFr['retinopathy'] == 0) | (mes.gtFr['retinopathy'] > 2)
    gtFr2 = mes.gtFr[lbl]
    X2 = X[lbl,:]
    y2 = (gtFr2['retinopathy']>0).values
    print 'n=', len(y2)
    crossValidation(X2, y2)
    #
    # print 'retinopathy 0 vs 1'
    # lbl = (mes.gtFr['retinopathy'] == 0) | (mes.gtFr['retinopathy'] == 1)
    # gtFr2 = mes.gtFr[lbl]
    # X2 = X[lbl,:]
    # y2 = (gtFr2['retinopathy']>0).values
    # print 'n=', len(y2)
    # crossValidation(X2, y2)
