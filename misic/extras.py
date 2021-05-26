import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize,rescale
from skimage.util import random_noise
from skimage.io import imread,imsave
from skimage.filters import gaussian
from skimage.feature import shape_index
from misic.utils import *
########## Suggested Pre and post processing functions here   #########
from skimage.segmentation import watershed
from scipy.ndimage import label
from scipy.ndimage import gaussian_laplace
from skimage.filters import gaussian,laplace
from skimage.exposure import adjust_gamma
from skimage.color import label2rgb

def postprocess_ws(im,yp):
    '''Watershed based postprocessing using image and its pixel probabilities'''
    # mask dilated
    mask = (yp[:,:,0] > 0.4)
    # watershed potential
    d = shape_index(im,1.5,mode = 'reflect')        
    
    # markers
    # get poles from contour predictions as markers
    sh = shape_index(yp[:,:,1],1,mode = 'reflect')
    markers,c = label(yp[:,:,0] > 0.95)
    # ther markers should be unique to each cell 
    markers = markers*(sh<-0.5)  # only poles    
    
    ws = watershed(d, markers=markers,watershed_line = True,mask = mask,compactness = 1,connectivity = 1)        
    return ws   

def add_noise(im,sensitivity = 0.1,invert = False,seed = 42):
    '''preprocessing by adding random noise to image only where the shape index map is around -0.4 for phase contrast images'''
    t0,s = -0.4,0.025        
    sim = -shape_index(im,1)
    if invert:
        sim = -sim
    ed = np.logical_and(sim > t0-s ,sim > t0+s)*1.0
    #ed = erosion(ed)
    np.random.seed(seed)
    noise = np.random.rand(ed.shape[0],ed.shape[1])
    noise = noise*(ed+0.5*np.random.rand(ed.shape[0],ed.shape[1]) )
    img = normalize2max(im) + sensitivity*noise
    return img