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







# def pre_processing(im,mean_width = 10):
#     scale = round(10.0/mean_width, 2)
#     if mean_width !=10:
#         im = rescale(im,scale)
#     #im = add_noise(im)
#     return im

def pre_processing(im,scale):    
    tmp = unsharp_mask(im*1.0)
    tmp = adjust_gamma(tmp,0.1)
    tmp = normalize2max(rescale(tmp,scale))    
    return noise_profile(tmp)    

def post_processing(y):
    yy = y[:,:,0] >0.95
    yy = y[:,:,0]-gaussian(y[:,:,1],0.5)
    yy = binary_opening(yy > 0.5)
    return yy  


class Microbenet():
    def __init__(self):
        self.size = 256
        model_path = get_file(
            'microbenet_model',
            'https://github.com/pswapnesh/Models/raw/master/MiSiDC04082020.h5') ## 0721
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
