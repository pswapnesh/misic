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
from MiSiC.utils import *
from scipy.ndimage import gaussian_laplace


# def pre_processing(im,scale):    
#     tmp = unsharp_mask(im*1.0)
#     tmp = adjust_gamma(tmp,0.1)
#     tmp = normalize2max(rescale(tmp,scale))    
#     return noise_profile(tmp)    

def pre_processing(im,scale,noise_var = 0.0001):    
    tmp = unsharp_mask(im*1.0)
    #tmp = adjust_gamma(tmp,0.25)    
    tmp = (rescale(im,scale))        
    tmp = gaussian_laplace(tmp,sigma = 1.8)
    tmp = 1-normalize2max(tmp)    
    return noise_profile(tmp,noise_var*scale)

def post_processing(y,orig_size):
    
    return resize(y[:,:,0],orig_size) >0.90
    

class MiSiC():
    def __init__(self):
        self.size = 256
        model_path = get_file(
            'misic_model',
            'https://github.com/pswapnesh/Models/raw/master/MiSiDC04082020.h5') ## 0721
        self.model = load_model(model_path,compile=False)
        self.model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    def shapeindex_preprocess(self,im):
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
        sh = self.shapeindex_preprocess(im)
        
        tiles,params = extract_tiles(sh,size = self.size,padding = 8)
        
        yp = self.model.predict(tiles)

        return stitch_tiles(yp,params)[pw:-pw,pw:-pw,:]
