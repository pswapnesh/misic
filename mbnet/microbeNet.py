from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file 
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize,rescale
from skimage.util import random_noise,pad
from skimage.io import imread,imsave
from skimage.filters import gaussian, laplace, threshold_otsu, median
from skimage.feature import shape_index
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from MicrobeNet.utils import *


def predict_small_images(im):
    sr,sc = im.shape
    sz = 256
    r = int(np.ceil(sz/sr))
    im = np.tile(im,(r,r))
    if im.shape[0]>sz:
        y = shnet.segment(im)  
        return y[:sr,:sc,:]
    else:
        x = shnet.shapenet_preprocess(im)
        y = shnet.unet.model.predict(x[np.newaxis,:])    
    return y[0,:sr,:sc,:]
        

def add_noise(im,sensitivity = 0.8):
    if sensitivity ==0:
        sensitivity = 0.1
    im = normalize2max(im)
    lvar = gaussian((im-gaussian(im,2))**2,2)
    lvar = (1.0-sensitivity)*lvar
    lvar[lvar<0.0001] = 0.0001
    return random_noise(im,mode = 'localvar',local_vars = lvar,clip=True,seed = 42)




def pre_processing(im,mean_width = 10):
    scale = round(10.0/mean_width, 2)
    im = rescale(im,scale)
    im = add_noise(im)
    return im

def post_processing(y):
    y = y[:,:,0] - y[:,:,1]
    y = 255.0*(y>0.95)
    return y    


class Microbenet():
    def __init__(self):
        self.size = 256
        model_path = get_file(
            'microbenet_model',
            'https://mycore.core-cloud.net/index.php/s/xwepbpNX1JH8hqL/download')        
        self.model = load_model(model_path,compile=False)
        self.model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    def shapenet_preprocess(self,im):
        sh = np.zeros((im.shape[0],im.shape[1],3))
        if np.max(im) ==0:
            return sh
        pw = 15
        im = pad(im,pw,'reflect')
        sh = np.zeros((im.shape[0],im.shape[1],3))    
        sh[:,:,0] = shape_index(im,1)
        sh[:,:,1] = shape_index(im,1.5)
        sh[:,:,2] = shape_index(im,2)     
        #sh = 0.5*(sh+1.0)
        return sh[pw:-pw,pw:-pw,:]
    
    def segment(self,im,invert = False):
        im = normalize2max(im)        
        pw = 16
        if invert:
            im = 1.0-im
        im = pad(im,pw,'reflect')
        sh = self.shapenet_preprocess(im)
        
        tiles,params = extract_tiles(sh,size = self.size,padding = 8)
        
        yp = self.model.predict(tiles)

        return stitch_tiles(yp,params)[pw:-pw,pw:-pw,:]
