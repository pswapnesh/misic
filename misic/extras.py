import os
import numpy as np
from skimage.transform import resize,rescale
from skimage.util import random_noise
from skimage.io import imread,imsave
from skimage.filters import gaussian
from skimage.feature import shape_index
from misic_ui.misic.utils import *
########## Suggested Pre and post processing functions here   #########
from skimage.segmentation import watershed
from scipy.ndimage import label
from scipy.ndimage import gaussian_laplace
from skimage.filters import gaussian,laplace
from skimage.exposure import adjust_gamma
from skimage.color import label2rgb
from skimage.morphology import remove_small_objects,remove_small_holes
from scipy.ndimage import label

def postprocess_ws(im,yp):
    '''Watershed based postprocessing using image and its pixel probabilities'''
    # mask dilated
    mask = (yp> 0.4)
    # watershed potential
    d = shape_index(im,1.5,mode = 'reflect')        
    
    # markers
    # get poles from contour predictions as markers
    
    markers,c = label(yp> 0.93)
    # ther markers should be unique to each cell     
    
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

def postprocessing(im,yp,mean_width=10,threshold_high= 0.999,threshold_low = 0.5):         
    markers = remove_small_objects(yp > threshold_high,0.5*mean_width**2)
    markers,c = label(markers)
    mask = yp > threshold_low
    mask = remove_small_holes(mask,0.1*mean_width**2)
    ws = watershed(im,markers = markers,mask = mask,watershed_line= True)

    return ws