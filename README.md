# MiSiC
### Microbe segmentation in dense colonies.

## Installation
Requires version python version 3.6/7

`pip install MiSiC`


## Usage

### use package
```python
from MiSiC.MiSiC import *
from skimage.io import imsave,imread
from skimage.transform import resize,rescale

filename = 'awesome_image.tif'

# read image using your favorite package
im = imread(filename)

# Parameters that need to be changed
## Ideally, use a single image to fine tune two parameters : mean_width and noise_variance (optional)

#input the approximate mean width of microbe under consideration
mean_width = 8

# compute scaling factor
scale = (10/mean_width)

# Initialize MiSiC
misic = MiSiC()

# preprocess using inbuit function or if you are feeling lucky use your own preprocessing
im = rescale(im,scale,preserve_range = True)

# add local noise
img = add_noise(im,sensitivity = 0.13,invert = True)

# segment
yp = misic.segment(img,invert = True)
yp = resize(yp,[sr,sc,-1])

# watershed based post processing
yp = postprocess_ws(img,yp)

# save 8-bit segmented image and use it as you like
imsave('segmented.tif', yp.astype(np.uint8))
''''

### In case of gpu error, one might need to disabple gpu before importing MiSiC [ os.environ["CUDA_VISIBLE_DEVICES"]="-1" ]