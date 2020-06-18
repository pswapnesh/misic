import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize,rescale
from skimage.util import random_noise,pad
#from skimage.external.tifffile import imread,imsave
from skimage.io import imread,imsave
from skimage.filters import gaussian, laplace, threshold_otsu, median
from skimage.feature import shape_index
from skimage.feature import hessian_matrix, hessian_matrix_eigvals



def normalize2max(im):
    im = im-np.min(im)
    return im/np.max(im)

def getPatch(im,sz):
    sr,sc = im.shape
    rr = np.random.randint(sr-sz)
    cc = np.random.randint(sc-sz)
    return im[rr:rr+sz,cc:cc+sz],rr,cc

from skimage.util import view_as_windows,pad
def get_padding(im,size = 256,stride = 256):
    sr,sc = im.shape[0],im.shape[1]    
    pad_r = stride-((sr-size)%stride)
    pad_c = stride-((sc-size)%stride)
    
    if (sr-size)%stride ==0:
        pad_r=0
    if (sc-size)%stride ==0:
        pad_c=0
    return int(pad_r),int(pad_c)

def extract_tiles(im,size=256,padding=16):     
    
    stride = size - 2*padding    
    if len(im.shape)<3:
        im = im[:,:,np.newaxis]
    sr,sc,ch = im.shape    
    
    pad_r,pad_c = get_padding(im,size,stride)
    
    im = pad(im,((0,pad_r),(0,pad_c),(0,0)),'reflect')
    patches = view_as_windows(im,(size,size,ch),stride)    
    patches = patches[:,:,0,:]
    
    sh = list(patches.shape)
    sh[1] = sh[0]*sh[1]
    sh = np.delete(sh,0)
    patches = np.reshape(patches,tuple(sh))
    
    R = np.arange(im.shape[0])
    rv = view_as_windows(R,size,stride) 
    rv = rv[:,0]
    
    C = np.arange(im.shape[1])
    cv = view_as_windows(C,size,stride) 
    cv = cv[:,0]
    cc,rr = np.meshgrid(cv,rv)
    positions = np.concatenate((rr.ravel()[:,np.newaxis],cc.ravel()[:,np.newaxis]),axis = 1)
    
    params = {}
    params['padding'] = padding
    params['pad_r'] = pad_r
    params['pad_c'] = pad_c
    params['im_size'] = im.shape[:2]
    params['positions'] = positions
    
    return patches,params

def stitch_tiles(patches,params):
    padding = params['padding']
    pad_r = params['pad_r']
    pad_c = params['pad_c']
    im_size = params['im_size']
    positions = params['positions']
    size = patches.shape[1]
    
    result = np.zeros((im_size[0],im_size[1],patches.shape[-1]))
    
    for i,pos in enumerate(positions):
        rr,cc = pos[0],pos[1]    
        result[rr:rr+size,cc:cc+size,:] += pad(patches[i,padding:-padding,padding:-padding,:],((padding,padding),(padding,padding),(0,0)),'constant')
    if pad_r>0:
        result = result[:-pad_r,:]
    if pad_c>0:
        result = result[:,:-pad_c]
    return result