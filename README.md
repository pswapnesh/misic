# MiSiC
### Agnostic segmentation of rod-shaped bacteria
Description - pending

## Installation
Requires version python version 3.6

pip install git+https://github.com/pswapnesh/MiSIC.git

or 

pip install https://github.com/pswapnesh/MiSiC/archive/master.zip


## Usage
### command line
mbnet --light_background True --mean_width 8 --src '/path/to/source/folder/\*.tif' --dst '/path/to/destination/folder/'

mbnet -lb True -mw 8 -s /path/to/source/folder/*.tif -d /path/to/destination/folder/

### use package
from MiSiC.MiSiC import *
from skimage.io import imsave,imread

misic = MiSiC()

im = imread(filename)

im = pre_processing(im,scale = 1)

y = misic.segment(im,invert = True) # invert = True for light backgraound images like Phase contrast

y = post_processing(y)

imsave('segmented.tif', (y*255).astype(np.uint8))

