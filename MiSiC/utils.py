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


def jaccard_coef(y_true, y_pred):
    smooth = 0.001
    #y_pred = K.cast(K.greater(y_pred, .8), dtype='float32') # .5 is the threshold
    #y_true = K.cast(K.greater(y_true, .9), dtype='float32') # .5 is the threshold
    intersection = np.mean(y_true * y_pred)
    sum_ = np.mean(y_true + y_pred)

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return jac

def get_rand_patch(im,sz):
    sr,sc = im.shape
    rr = np.random.randint(sr-sz)
    cc = np.random.randint(sc-sz)
    return im[rr:rr+sz,cc:cc+sz]

def find_best_parameter(im,mbnet,scale=1,invert = True):
    sr,sc = im.shape
    N = 5
    if scale != 1:
        im = rescale(im,scale)
        
    variances = np.arange(0.0,0.01,0.0005)
    imgs = []
    for v in variances:    
        if v ==0:
            r1 = im*1.0
        else:
            r1 = random_noise(im,mode = 'gaussian',var = v,seed = 42)
        r1 = normalize2max(r1)
        if invert:
            r1 = 1.0-r1
        random_patches = np.array([mbnet.shapenet_preprocess(get_rand_patch(r1,256)) for ii in range(N)])            
        y1 = mbnet.model.predict(random_patches)        
        imgs.append(y1)
    
    imgs = np.array(imgs)

    ym = np.mean(1.0*(imgs>0.98),axis = 0)
    J = np.array([jaccard_coef(i,ym) for i in imgs])
    idx = np.where(J == np.max(J))[0][0]    
    return variances[idx],[variances,J]
        
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
        

# def unsharp_mask(im):
#     return im - 0.8*gaussian(laplace(im),2)

def unsharp_mask(im,c=0.6,sigma=1):
    return (c/(2*c-1))*im - (1-c)/(2*c - 1)*gaussian(im,sigma)

def noise_profile(im,var1 = 0.005):    
    im = normalize2max(im)
    gr,gc = np.gradient(im)
    e = gaussian(np.sqrt(gr**2 + gc**2),0.5)
    #e = np.abs(laplace(gaussian(im,2)))
    e = normalize2max(e)
    return random_noise(im,mode = 'localvar',local_vars = 0.00001+var1*(1-e),seed = 42)
    