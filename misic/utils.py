import numpy as np
from skimage.util import view_as_windows


def normalize2max(im):
    ''' normalize to max '''
    if np.max(im) == 0:
        return im
    im = im-np.min(im)
    return im/np.max(im)

def getPatch(im,sz):
    ''' get random patch from image of size szXsz '''
    sr,sc = im.shape
    rr = np.random.randint(sr-sz)
    cc = np.random.randint(sc-sz)
    return im[rr:rr+sz,cc:cc+sz],rr,cc


def get_coords(sr,sc,size=256,exclude = 16):
    rr = np.arange(0,sr,size-2*exclude)
    d = (size-2*exclude) - sr%(size-2*exclude)
    rr[-1] = sr-(size-2*exclude) 
    cc = np.arange(0,sc,size-2*exclude)
    d = (size-2*exclude) - sc%(size-2*exclude)
    cc[-1] =sc -(size-2*exclude) 
    rr,cc = np.meshgrid(rr,cc)
    rr = rr.ravel()
    cc = cc.ravel()
    return rr,cc



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



# def tile_generator(im,size = 256, exclude = 16):
#     sr,sc,_ = im.shape
#     rr,cc = get_coords(sr,sc,size,exclude)
#     def generator():
#         for r,c in zip(rr,cc):
#             r0,r1 = r-exclude,r-exclude+size
#             c0,c1 = c-exclude,c-exclude+size    
#             if (r0 > 0) & (r1 < sr) & (c0 > 0) & (c1 < sc):
#                 tmp = im[r0:r1,c0:c1,:]
#                 # top
#             elif (r0 < 0) & (r1 < sr) & (c0 > 0) & (c1 < sc):
#                 tmp = np.pad(im[r:r1,c0:c1,:],((-r0,0),(0,0),(0,0)),'reflect')
#                 # bottom
#             elif (r0 > 0) & (r1 > sr) & (c0 > 0) & (c1 < sc):
#                 tmp = np.pad(im[r0:sr,c0:c1,:],((0,r1-sr),(0,0),(0,0)),'reflect')
#                 # left
#             elif (r0 > 0) & (r1 < sr) & (c0 < 0) & (c1 < sc):
#                 tmp = np.pad(im[r0:r1,c:c1,:],((0,0),(-c0,0),(0,0)),'reflect')
#                 # right
#             elif (r0 > 0) & (r1 < sr) & (c0 > 0) & (c1 > sc):
#                 tmp = np.pad(im[r0:r1,c0:sc,:],((0,0),(0,c1-sc),(0,0)),'reflect')    
#                 # top left
#             elif (r0 < 0) & (r1 < sr) & (c0 < 0) & (c1 < sc):
#                 tmp = np.pad(im[r:r1,c:c1,:],((-r0,0),(-c0,0),(0,0)),'reflect')    
#                 # top right
#             elif (r0 < 0) & (r1 < sr) & (c0 > 0) & (c1 > sc):
#                 tmp = np.pad(im[r:r1,c0:sc,:],((-r0,0),(0,c1-sc),(0,0)),'reflect') 
#                 # bottom left
#             elif (r0 > 0) & (r1 > sr) & (c0 < 0) & (c1 < sc):
#                 tmp = np.pad(im[r0:,c:c1,:],((0,r1-sr),(-c0,0),(0,0)),'reflect') 
#                 # bottom right
#             elif (r0 > 0) & (r1 > sr) & (c0 > 0) & (c1 > sc):
#                 tmp = np.pad(im[r0:sr,c0:sc,:],((0,r1-sr),(0,c1-sc),(0,0)),'reflect') 
#                 # smaller image
#             elif (r0 < 0) & (r1 > sr) & (c0 < 0) & (c1 > sc):
#                 tmp = np.pad(im,((-r0,r1-sr),(-c0,c1-sc),(0,0)),'reflect') 
#             yield tmp
#     return generator
        

# def stitch_tiles(res,gen):
#     sr,sc , _ = res.shape
#     rr,cc = get_coords(sr,sc,size,exclude)
#     for r,c,img in zip(rr,cc,gen):
#         r0,r1 = r-exclude,r-exclude+size
#         c0,c1 = c-exclude,c-exclude+size    
#         #print(r0,r1,c0,c1)
#         if (r0 > 0) & (r1 < sr) & (c0 > 0) & (c1 < sc):
#             res[r:r-2*exclude+size,r:r-2*exclude+size,:] = img[exclude:-exclude,exclude:-exclude,:]
#             # top
#         elif (r0 < 0) & (r1 < sr) & (c0 > 0) & (c1 < sc):
#             res[r:r+size-2*exclude,c:c+size-2*exclude,:] = img[exclude:-exclude,exclude:-exclude,:]
#             # bottom
#         elif (r0 > 0) & (r1 > sr) & (c0 > 0) & (c1 < sc):
#             res[r:sr,c:c+size-2*exclude,:] = img[exclude:-(r1-sr),exclude:-exclude,:]
#             # left
#         elif (r0 > 0) & (r1 < sr) & (c0 < 0) & (c1 < sc):
#             res[r:r+size-2*exclude,c:c+size-2*exclude,:] = img[exclude:-exclude,exclude:-exclude,:]
#             # right
#         elif (r0 > 0) & (r1 < sr) & (c0 > 0) & (c1 > sc):
#             res[r:r+size-2*exclude,c:c+size-2*exclude,:] = img[exclude:-exclude,exclude:-(c1-sc),:]
#             # top left
#         elif (r0 < 0) & (r1 < sr) & (c0 < 0) & (c1 < sc):
#             res[r:r+size-2*exclude,c:c+size-2*exclude,:] = img[exclude:-exclude,exclude:-exclude,:]
#             # top right
#         elif (r0 < 0) & (r1 < sr) & (c0 > 0) & (c1 > sc):
#             res[r:r+size-2*exclude,c:sc,:] = img[exclude:-exclude,exclude:-(c1-sc),:]
#             # bottom left
#         elif (r0 > 0) & (r1 > sr) & (c0 < 0) & (c1 < sc):      
#             res[r:,c:c+size-2*exclude,:] = img[exclude:-(r1-sr),exclude:-exclude,:]
#             # bottom right
#         elif (r0 > 0) & (r1 > sr) & (c0 > 0) & (c1 > sc):
#             res[r:sr,c:sc,:] = img[exclude:-(r1-sr),exclude:-(c1-sc),:]
#             # smaller image
#         elif (r0 < 0) & (r1 > sr) & (c0 < 0) & (c1 > sc):
#             res = img[r0:-(r1-sr),c0:-(c1-sc),:]
