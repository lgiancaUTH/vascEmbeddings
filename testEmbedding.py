import sys
# include subdirectories
sys.path.insert(0, './lib/')
sys.path.insert(0, './src/')

import numpy as np
import matplotlib.pyplot as plt
import keras as k
import tensorflow as tf
assert k.backend.image_dim_ordering() == 'tf'

import skimage.io as skio
import skimage.transform as sktr

import segmentVasculature as sv


if __name__ == '__main__':
    # load example image
    imgStr = 'data/' + '20051020_55701_0100_PP.tif'
    img = skio.imread(imgStr)

    # create embedding in one step
    embVec, resEncCoordKept = sv.createOMIAembedding( img )

    #or run a visual test to see segmentation driving the embeddings
    #===========================================================
    # load model segmentation
    model = sv.getTrainedModel()

    # create model encoding output
    modelEnc =  sv.genEncModel()
    if modelEnc is  None:
        print 'encoding layer not available'

    # segment
    resImg =  sv.segmentImage( img, model )
    #encoding (if available)
    if modelEnc is not None:
        resEnc, resEncCoord =  sv.getImageEncoding( img, modelEnc )

    # visual output
    skio.imshow(  np.hstack( (sktr.resize(img[:,:,1], resImg.shape), resImg) ) )

    sv.checkFiltering(resImg, resEnc, resEncCoord)
    plt.show()
    # ===========================================================

