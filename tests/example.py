from misic.misic import *
from misic.extras import *
from skimage.io import imsave,imread
from skimage.transform import resize,rescale

filename = 'awesome_image.tif'

# read image using your favorite package
im = imread(filename)
sr,sc = im.shape

# Parameters that need to be changed
## Ideally, use a single image to fine tune two parameters : mean_width and noise_variance (optional)

#input the approximate mean width of microbe under consideration
standard_width = 9.7

# the approximate width of cells to be segmented
mean_width = 9.7

# If image is phase contrast light_background = True
light_background = True

# compute scaling factor
scale = (standard_width/mean_width)

# Initialize MiSiC
mseg = MiSiC()

# preprocess using inbuit function or if you are feeling lucky use your own preprocessing
# recomended preprcessing
# im = adjust_gamma(im,0.25)
# im = unsharp_mask(im,2.2,0.6)

# for fluorescence images
# im = gaussian(laplace(im),2.2)
# im = add_noise(im,0.1)
# OR
# im = random_noise(im,mode = 'gaussian',var = 0.1/100.0)

im = rescale(im,scale,preserve_range = True)

# add local noise
img = add_noise(im,sensitivity = 0.13,invert = light_background)

# segment
yp = mseg.segment(img,invert = light_background)
yp = resize(yp,(sr,sc))

# watershed based post processing (optional)
# yp = postprocess_ws(img,yp)
yp = postprocessing(im if light_background else -im,yp)[:,:,0]  

# save 8-bit segmented image and use it as you like
imsave('segmented.tif', ((yp > 0)*255).astype(np.uint8))