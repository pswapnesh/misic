import numpy as np
from skimage.util import view_as_windows


def normalize2max(im):
    ''' normalize to max '''
    im = im-np.min(im)
    return im/np.max(im)

def getPatch(im,sz):
    ''' get random patch from image of size szXsz '''
    sr,sc = im.shape
    rr = np.random.randint(sr-sz)
    cc = np.random.randint(sc-sz)
    return im[rr:rr+sz,cc:cc+sz],rr,cc



def extract_tiles(im,size = 512,exclude = 12):
    ''' extract tiles from image of size 'size' to be stiched back such that 'exclude' pixels near border of tile are excluded '''
    size = size-2*exclude

    if len(im.shape)<3:
        im = im[:,:,np.newaxis]
    sr,sc,ch = im.shape 
    
    pad_row = 0 if sr%size == 0 else (int(sr/size)+1) * size - sr
    pad_col = 0 if sc%size == 0 else (int(sc/size)+1) * size - sc
    im1 = np.pad(im,((0,pad_row),(0,pad_col),(0,0)),mode = 'reflect')
    sr1,sc2,_ = im1.shape
    

    rv = np.arange(0,im1.shape[0],size)
    cv = np.arange(0,im1.shape[1],size)
    cc,rr = np.meshgrid(cv,rv)
    positions = np.concatenate((rr.ravel()[:,np.newaxis],cc.ravel()[:,np.newaxis]),axis = 1)
        
    im1 = np.pad(im1,((exclude,exclude),(exclude,exclude),(0,0)),mode = 'reflect')

    params = {}
    params['size'] = size
    params['exclude'] = exclude
    params['pad_row'] = pad_row
    params['pad_col'] = pad_col
    params['im_size'] = [sr1,sc2]
    params['positions'] = positions
    
    patches = view_as_windows(im1,(size+2*exclude,size+2*exclude,ch),size)    
    patches = patches[:,:,0,:,:,:]
    patches = np.reshape(patches,(-1,patches.shape[2],patches.shape[3],patches.shape[4]))
    return patches,params


def stitch_tiles(patches,params):
    ''' stitch tiles generated from extract tiles '''
    size = params['size']
    pad_row = params['pad_row']
    pad_col = params['pad_col']
    im_size = params['im_size']
    positions = params['positions']
    exclude = params['exclude']
        
    
    result = np.zeros((im_size[0],im_size[1],patches.shape[-1]))*1.0
    
    
    for i,pos in enumerate(positions):        
        rr,cc = pos[0],pos[1]    
        result[rr:rr+size,cc:cc+size,:] = patches[i,exclude:-exclude,exclude:-exclude,:]*1.0
    
    if pad_row>0:
        result = result[:-pad_row,:]
    if pad_col>0:
        result = result[:,:-pad_col]
    return result