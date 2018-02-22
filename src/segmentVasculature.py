###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
##################################################

#Python
import numpy as np
import ConfigParser
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
import sys

import skimage.io as skio
import skimage.transform as sktr

from skimage.filters import  threshold_otsu

sys.path.insert(0, './lib/')


# help_functions.py
from help_functions import *


# pre_processing.py
from pre_processing import my_PreProc


#========= CONFIG FILE TO READ FROM =======
config = ConfigParser.RawConfigParser()
# config.read('test3/test3_configuration.txt') # original
config.read('test6/test6_configuration.txt') # with encoding
#===========================================
#run the training on invariant or local
path_data = config.get('data paths', 'path_local')

#original images size used for training (for FOV selection - DRIVE)
full_img_height = 584
full_img_width = 565

# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
#the stride in case output with average
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)
#model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment +'/'


#N full images to be predicted
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
#Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))
#====== average mode ===========
average_mode = config.getboolean('testing settings', 'average_mode')


# Image resolution for the dataset used for training
NET_RES = [full_img_height,full_img_width]



def extract_ordered_overlap(full_imgs, patch_h, patch_w,stride_h,stride_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]
    print "Number of patches on h : " +str(((img_h-patch_h)//stride_h+1))
    print "Number of patches on w : " +str(((img_w-patch_w)//stride_w+1))
    print "number of patches per image: " +str(N_patches_img) +", totally for this dataset: " +str(N_patches_tot)
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
    coordArr = np.empty(  (N_patches_tot, 4), dtype=int ) # coordinates fpr patches x (y,x,height,width)
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                y = h*stride_h
                height = (h*stride_h)+patch_h
                x= w*stride_w
                width = (w*stride_w)+patch_w
                # extract patch
                patch = full_imgs[i, :, y:height, x:width]
                # patch = full_imgs[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                patches[iter_tot] = patch
                # add coordinates used
                coordArr[iter_tot,:] = [y,x,height,width]
                # increment
                iter_tot +=1
    assert (iter_tot==N_patches_tot)
    return patches, coordArr  #array with all the full_imgs divided in patches



def recompone_overlapTF(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape)==3)  #3D arrays
    # assert (preds.shape[1]==1 or preds.shape[1]==3)  #check the channel is 1 or 3
    patch_h = preds.shape[1]
    patch_w = preds.shape[2]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    print "N_patches_h: " +str(N_patches_h)
    print "N_patches_w: " +str(N_patches_w)
    print "N_patches_img: " +str(N_patches_img)
    assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    print "According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)"
    full_prob = np.zeros((N_full_imgs,1,img_h,img_w))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs,1,img_h,img_w))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k]
                full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1
                k+=1
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
    print final_avg.shape
    assert(np.max(final_avg)<=1.0) #max value for a pixel is 1.0
    assert(np.min(final_avg)>=0.0) #min value for a pixel is 0.0
    return final_avg


def segmentImage(imgIn, modelIn):
    USE_OVERLAP = 1 # overlap 0 not working
    #

    # resize according to DRIVE set using height as reference
    newWidth = int(NET_RES[0] * 1.0 / imgIn.shape[0] * imgIn.shape[1])
    img = sktr.resize(imgIn, (NET_RES[0], newWidth) )

    # trim to make it compatible with patches
    pxWidth = img.shape[1] - ( (img.shape[1] - patch_width) % stride_width )
    pxHeight = img.shape[0] - ( (img.shape[0] - patch_height) % stride_height )
    img = img[0:pxHeight, 0:pxWidth]
    # make it compatible with pipeline
    # add axis
    img2 = img[ np.newaxis, :, :, : ]
    # move channel to second dimension
    img2 = img2.transpose(0, 3, 1, 2)
    img3 = my_PreProc(img2)

    # extract patches
    # array (patch number x 1 x patch_height x patch_width)
    patchesImgs,_ = extract_ordered_overlap(img3, patch_height, patch_width, stride_height, stride_width)
    # convert to tf
    patches_imgs_test = patchesImgs.transpose( 0, 2, 3, 1 )
    #Calculate the predictions (returns numPatches x patchPx x 2)
    predictions = modelIn.predict(patches_imgs_test, batch_size=32, verbose=2)
    #---- back to patches
    #(speeded up version of pred_to_imgs(predictions, "original"))
    patchesArr = predictions[:,:,1] # vessel probability channel
    patchesArr2 = np.reshape(patchesArr, (patchesArr.shape[0], patch_height, patch_width))
    #---

    # compose image again
    predImg = recompone_overlapTF(patchesArr2, img.shape[0], img.shape[1], stride_height, stride_width )

    return predImg[0,0,:,:]


def getImageEncoding(imgIn, modelIn):
    """
    Get vessel based image encodings (model from @getTrainedModel )

    :param imgIn:
    :param modelIn:
    :return:
    """
    # resize according to DRIVE set using height as reference
    newWidth = int(NET_RES[0] * 1.0 / imgIn.shape[0] * imgIn.shape[1])
    img = sktr.resize(imgIn, (NET_RES[0], newWidth) )

    # trim to make it compatible with patches
    pxWidth = img.shape[1] - ( (img.shape[1] - patch_width) % stride_width )
    pxHeight = img.shape[0] - ( (img.shape[0] - patch_height) % stride_height )
    img = img[0:pxHeight, 0:pxWidth]
    # make it compatible with pipeline
    # add axis
    img2 = img[ np.newaxis, :, :, : ]
    # move channel to second dimension
    img2 = img2.transpose(0, 3, 1, 2)
    img3 = my_PreProc(img2)

    # extract patches
    # array (patch number x 1 x patch_height x patch_width)
    patchesImgs, patchesCoord = extract_ordered_overlap(img3, patch_height, patch_width, stride_height, stride_width)
    # convert to tf
    patches_imgs_test = patchesImgs.transpose( 0, 2, 3, 1 )
    #Calculate the predictions (returns numPatches x patchPx x 2)
    predictions = modelIn.predict(patches_imgs_test, batch_size=32, verbose=2)

    return predictions, patchesCoord

def getTrainedModel():
    """
    Load existing NN model return it
    :return: existing NN model
    """
    best_last = 'best' # or last
    modelTmp = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
    modelTmp.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')

    return modelTmp

def genEncModel(innerLayerName='fullyEncoded'):
    """
    Load existing NN model and set inner layer as output for performing the transfer learning

    :param innerLayerName: name of the inner layer to perform the transfer learning
    :return: NN model or none if layer not available
    """

    # get no encoded model
    modelNoEnc = getTrainedModel()
    # get layer and set new output
    encodedLayer = modelNoEnc.get_layer( innerLayerName )
    modelEnc = None
    if encodedLayer is not None:
        modelEnc = Model(inputs=modelNoEnc.input, outputs=encodedLayer.output)

    return modelEnc


def checkFiltering( resImg, resEnc, resEncCoord ):
    """
    Check filtering used in tranfLearning function
    :param resImg:
    :param resEnc:
    :param resEncCoord:
    :return:
    """

    # -- Check filter location
    coeffSum = np.sum(np.abs(resEnc), axis=1)  # sum across vectors
    th = threshold_otsu(coeffSum)
    resEncCoordKept = resEncCoord[coeffSum < th, :]


    coordImg = np.zeros( resImg.shape, dtype=np.float)
    for patchId in range( resEncCoordKept.shape[0] ):
        y = resEncCoordKept[patchId,0]
        x = resEncCoordKept[patchId, 1]
        height = resEncCoordKept[patchId, 2]
        width = resEncCoordKept[patchId, 3]

        coordImg[y:height,x:width] = 1

    plt.figure()
    skio.imshow( np.hstack( (coordImg, resImg) )  )
    # -


def createOMIAembedding( imgIn ):
    """
    Create OMIA embedding from a colour image matrix

    :param imgIn: image array
    :param modelIn: trained model
    :return: (embedding vector, locations used)
    """

    modelEnc = genEncModel()
    if modelEnc is  None:
        print 'encoding layer not available'
        return None

    # get embedding according to default model
    resEnc, resEncCoord = getImageEncoding(imgIn, modelEnc)

    # compute mask automatically
    coeffSum = np.sum(np.abs(resEnc), axis=1)  # sum across vectors
    th = threshold_otsu(coeffSum)
    #

    # keep only embeddings inside the mask
    resEncCoordKept = resEncCoord[coeffSum < th, :]
    resEnc = resEnc[coeffSum < th, :]
    #

    # Compute statistics on the embeddings
    q75, q50, q25, q1, q99 = np.percentile(resEnc, [75, 50, 25, 1, 99], axis=0, keepdims=False)
    iqr = q75 - q25

    # create embedding (i.e. feature vector)
    statFeatVec = np.append(q50, iqr)


    return (statFeatVec, resEncCoordKept)


if __name__ == '__main__':


    # load model segmentation
    model = getTrainedModel()

    # create model encoding output
    modelEnc = genEncModel()
    if modelEnc is  None:
        print 'encoding layer not available'

    # load example image
    imgStr = 'data/' + '20051020_55701_0100_PP.tif'
    img = skio.imread(imgStr)

    # segment
    resImg = segmentImage( img, model )
    #encoding (if available)
    if modelEnc is not None:
        resEnc, resEncCoord = getImageEncoding( img, modelEnc )

    # visual output
    skio.imshow(  np.hstack( (sktr.resize(img[:,:,1], resImg.shape), resImg) ) )

    checkFiltering(resImg, resEnc, resEncCoord)
    plt.show()
    #

    # create embedding in one step
    embVec, resEncCoordKept = createOMIAembedding( img )



