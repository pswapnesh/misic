from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file 
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize,rescale
from skimage.util import random_noise
from skimage.io import imread,imsave
from skimage.filters import gaussian
from skimage.feature import shape_index
from MiSiC.utils import *

class MiSiC():
    def __init__(self,model_name = 'MiSiC/MiSiDC04082020.h5'):
        model_path = get_file(
            'misic_model',
            'https://github.com/pswapnesh/Models/raw/master/MiSiDC04082020.h5') ## 0721
        self.model = load_model(model_path,compile=False)
        self.model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        self.size = 256        
        
    def shapeindex_preprocess(self,im):
        ''' apply shap index map at three scales'''
        sh = np.zeros((im.shape[0],im.shape[1],3))
        if np.max(im) ==0:
            return sh
        
        # pad to minimize edge artifacts
        pw = 8
        im = np.pad(im,pw,'reflect')            
        sh[:,:,0] = shape_index(im,1)[pw:-pw,pw:-pw]
        sh[:,:,1] = shape_index(im,1.5)[pw:-pw,pw:-pw]
        sh[:,:,2] = shape_index(im,2)[pw:-pw,pw:-pw]
        #sh = 0.5*(sh+1.0)
        return sh
    
    def segment(self,im,invert = False,exclude = 32):        
        sh = self.shapeindex_preprocess(im)        
        sh = -sh if invert else sh
        tiles,params = extract_tiles(sh,size = self.size,exclude = exclude)               
        yp = self.model.predict(tiles)
        return stitch_tiles(yp,params)
