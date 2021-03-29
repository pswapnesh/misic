# MiSiC
### Microbe segmentation in dense colonies.

## Installation
Requires version python version 3.6 or 3.7

`pip install TOBEFILLED`

## Usage

### use package
```python
from MiSiC.MiSiC import *
from skimage.io import imsave,imread

filename = 'awesome_image.tif'

# read image using your favorite package
im = imread(filename)

# Parameters that need to be changed
## Ideally, use a single image to fine tune two parameters : mean_width and noise_variance (optional)

#input the approximate mean width of microbe under consideration
mean_width = 8
noise_variance = 0.0001

# compute scaling factor
scale = (10/mean_width)

# Initialize MiSiC
misic = MiSiC()

# preprocess using inbuit function or if you are feeling lucky use your own preprocessing
im = pre_processing(im,scale = scale, noise_var = noise_variance)

# segment the image with invert = True for light backgraound images like Phase contrast
y = misic.segment(im,invert = True)

# if you need both the body y[:,:,0] and contour y[:,:,1] skip the post processing.
y = post_processing(y,im.shape)

# save 8-bit segmented image and use it as you like
imsave('segmented.tif', (y*255).astype(np.uint8))

