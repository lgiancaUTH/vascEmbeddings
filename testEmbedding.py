from __future__ import print_function
import sys
# include subdirectories
sys.path.insert(0, './lib/')
sys.path.insert(0, './src/')

import time
import numpy as np
import matplotlib.pyplot as plt
import keras as k
import tensorflow as tf
assert k.backend.image_dim_ordering() == 'tf'

import skimage.io as skio
import skimage.transform as sktr

import segmentVasculature as sv


if __name__ == '__main__':
    VISUAL_OUTPUT = False


    # load example image
    imgStr = 'data/' + '20051020_55701_0100_PP.tif'
    img = skio.imread(imgStr)

    t0 = time.time()
    # create embedding in one step
    embVec, resEncCoordKept = sv.createOMIAembedding( img )
    t1 = time.time()

    total = t1-t0
    print ("-"*20,"Embeddings in ", total, " sec")

    #or run a visual test to see segmentation driving the embeddings
    #===========================================================
    # load model segmentation
    model = sv.getTrainedModel()

    # create model encoding output
    modelEnc =  sv.genEncModel()
    if modelEnc is  None:
        print ('encoding layer not available')

    # segment
    resImg =  sv.segmentImage( img, model )
    #encoding (if available)
    if modelEnc is not None:
        resEnc, resEncCoord =  sv.getImageEncoding( img, modelEnc )

    # visual output
    imgOut = np.hstack((sktr.resize(img[:, :, 1], resImg.shape), resImg))
    if VISUAL_OUTPUT:
        skio.imshow( imgOut  )

        sv.checkFiltering(resImg, resEnc, resEncCoord)
        plt.show()
    else:
        skio.imsave( 'data/testEmbOutput.png', imgOut )

    # ===========================================================

