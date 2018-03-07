###################################################
#
#   Script to
# test the iage retrieval with Messidor2 dataset
##################################################

#Python
from __future__ import print_function
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
import transfLearning as tl


sys.path.insert(0, './lib/')


class Messidor2:
    """
    Messidor2 dataset
    """
    def __init__(self):
        # File location
        self.baseDir = '/data/bigdata/retina/messidor2/'
        self.gtFile = self.baseDir + 'messidor-2.csv'
        self.imgDir = self.baseDir + 'IMAGES/'

        # load GT (and rename columns)
        self.gtOrigFr = pd.read_csv( self.gtFile )
        self.gtOrigFr.columns = pd.Index(['imgR', 'imgL'])
        # add pID column
        self.gtOrigFr['pID'] = range(len(self.gtOrigFr))

        #--- convert to one line per image (compatible with Messidor class)
        # right
        gtFr = self.gtOrigFr[['imgR', 'pID']].copy()
        gtFr.columns = pd.Index(['img', 'pID'])
        gtFr['side'] = 'r'
        # left
        gtFrTmp = self.gtOrigFr[['imgL', 'pID']].copy()
        gtFrTmp.columns = pd.Index(['img', 'pID'])
        gtFrTmp['side'] = 'l'
        # join
        self.gtFr = gtFr.append(gtFrTmp)
        # recalculate index
        self.gtFr = self.gtFr.reset_index(drop=True)
        # ---

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
    
    def viewPat(self, patID):
        """
        view image pair
        :param patID: numerical id
        :return: None
        """
        imgId1 = self.gtFr[self.gtFr['pID']==patID].index.values[0]
        imgId2 = self.gtFr[self.gtFr['pID']==patID].index.values[1]
        # load images
        info = self.loadInfo(imgId1)
        im1 = skio.imread( self.imgDir + info['img'])
        info = self.loadInfo(imgId2)
        im2 = skio.imread( self.imgDir + info['img'])
        
        #view
        plt.figure()
        skio.imshow( np.hstack( (im1,im2) ) )
        plt.axis('off') 


def findLRmatch( mesIn, X, nNeighMaxIn=5, stepIn=10 ):
    """
    Find similar images 'left right' receive Messidor as input
    :param mesIn:
    :param X:
    :param nNeighMaxIn: maximum number of neighbours
    :return: (neighboursLst, performanceLst) a list of number of neighbours and correctly identified pairs based on the neighbours
    """
    HEIGHT_VIS = 700
    N_NEIGH = nNeighMaxIn # neighbour comparison

    # pMod = man.TSNE(n_components=2, random_state=0)
    # X = pre.StandardScaler().fit_transform( X )
    # Xp = pMod.fit_transform(X)
    Xp = X

    # absolute path
    imgDir  = mesIn.imgDir
    # GT
    gtFr = mesIn.gtFr

    # Sides
    gtLFr = gtFr[gtFr['side']=='l'].reset_index(drop=True)
    gtRFr = gtFr[gtFr['side']=='r'].reset_index(drop=True)
    XpL = Xp[(gtFr['side']=='l'),:]
    XpR = Xp[(gtFr['side']=='r'),:]

    # fit nearest neighbor model
    nbrsMod = nei.NearestNeighbors(n_neighbors=N_NEIGH, algorithm='ball_tree').fit(XpL)
    # find closest image fomr the right eye set
    distVec, indDstVec = nbrsMod.kneighbors(XpR)
    
    # init a list of number of neighbours and correctly identified pairs based on the neighbours
    nnLst = []
    ratioFoundLst = []
    # for an increasing number of neighbours
    for curNN in range(1,nNeighMaxIn+1,stepIn):
        foundCnt = 0
        for i in range(len(gtRFr)):
            # find left target image from right
            refPiD =  gtRFr.iloc[i].pID
            lImgId = gtLFr[gtLFr['pID']==refPiD].index.values[0]
            # check if found
            found = (lImgId in indDstVec[i,:curNN])
            if found:
                foundCnt += 1
                
        ratioFound = (foundCnt*1.0)/len(gtRFr)
        
        # add to lists
        ratioFoundLst.append(ratioFound)
        nnLst.append(curNN)
        # print progress
        print (curNN, 'of', nNeighMaxIn)
        
            
        
    return (np.array(nnLst),np.array(ratioFoundLst)) 


if __name__ == '__main__':
    ENC_FILE = 'data/mess2-encVess.h5'
    FEAT_MAT_FILE = 'data/mess2-featMatSmStat.h5'

    # ENC_FILE = 'data/mess2-inv-encVess.h5'
    # FEAT_MAT_FILE = 'data/mess2-inv-featMatSmStat.h5'


    mes = Messidor2()


    # generate encoding vectors
    X = []
    if not os.path.exists(FEAT_MAT_FILE):
        # generate encoding vectors not flipped
        X = tl.generateEncoding(mes, ENC_FILE)
        tl.saveH5(X, 'featMat', FEAT_MAT_FILE)
        # #---- generate encoding vectors flipped
        # flippedArr = (mes.gtFr['side']=='r').values
        # X = tl.generateEncoding(mes, ENC_FILE, flippedArr)
        # tl.saveH5(X, 'featMat', FEAT_MAT_FILE)
        # #----
    else:
        print ('loading ', FEAT_MAT_FILE)
        X = tl.loadH5('featMat', FEAT_MAT_FILE)


    print (findLRmatch( mes, X, 1 ))


    #=== plot curve
    totPairs = len(mes.gtFr) / 2 
    nnArr, ratioArr = findLRmatch( mes, X, 50, 1 )
    # calculate chance
    chanceArr = (nnArr*1.) / totPairs
    
    plt.figure()
    plt.plot( nnArr, ratioArr, label='vasculature embeddings' )
    plt.plot( nnArr, chanceArr, label='retrieval by chance' )
    plt.ylabel('ratio of correctly matched pair')
    plt.xlabel('nearest neighbours needed to find the match')
    plt.title('Unsupervised Matching of Right/Left Retinas (n='+str(totPairs)+')')
    plt.legend()
    plt.show()
    #===
    

    # tmpPlotInteractiveProj(mes.gtFr, X, isInteractive=False)
    # findSimilarImgs(mes.gtFr, X, 5) # paper

