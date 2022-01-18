from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file 
import os
import numpy as np
from skimage.feature import shape_index
from misic_ui.misic.utils import *
from misic_ui.misic.extras import *
import tensorflow as tf


class MiSiC():
    def __init__(self):        
        # try:
        #     model_path = get_file('misic_model','https://github.com/pswapnesh/Models/raw/master/MiSiDC04082020.h5') ## 0721
        #     self.model = load_model(model_path,compile=False)            
        # except:
        model_name = os.path.join(os.path.dirname(__file__), 'MiSiCv2.h5')
        self.model = load_model(model_name,compile=False)

        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        self.size = 256        
        
    def shapeindex_preprocess(self,im):
        ''' apply shap index map at three scales'''
        sh = np.zeros((im.shape[0],im.shape[1],3))
        if np.max(im) ==0:
            return sh
        
        # pad to minimize edge artifacts                    
        sh[:,:,0] = shape_index(im,1, mode='reflect')
        sh[:,:,1] = shape_index(im,1.5, mode='reflect')
        sh[:,:,2] = shape_index(im,2, mode='reflect')
        return sh
    
    # def segment(self,im,invert = False,exclude = 16):        
    #     sh = self.shapeindex_preprocess(im)        
    #     sh = -sh if invert else sh
    #     batch_size = 16
    #     #tiles,params = extract_tiles(sh,size = self.size,exclude = exclude) 
    #     tiles = tf.data.Dataset.from_generator(tile_generator(self..shapeindex_preprocess(sh)),(tf.float32)).batch(batch_size)
    #     yp = self.model.predict(tiles)              
    #     return stitch_tiles(yp,yp)

    def segment(self,im,invert = False,exclude = 16):        
        #sh = self.shapeindex_preprocess(im)        
        #sh = -sh if invert else sh
        tiles,params = extract_tiles(im[:,:,np.newaxis],size = self.size,exclude = exclude)               
        tiles = np.array([self.shapeindex_preprocess(normalize2max(t))  for t in tiles[:,:,:,0]])
        tiles = -tiles if invert else tiles
        yp = self.model.predict(tiles)
        return stitch_tiles(yp,params)

    # def segment_auto(self,im,invert = False):
    #     im1 = add_noise(im,sensitivity = 0.1,invert = invert,seed = 42)
    #     y = self.segment(im1,invert = invert)
    #     return postprocess_ws(im,y)
    