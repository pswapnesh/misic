# MicrobeNet
### Agnostic segmentation of rod-shaped bacteria

## Installation
pip install git+https://github.com/pswapnesh/MicrobeNet.git


## Usage
### command line
mbnet --light_background True --mean_width 8 --src /path/to/source/folder/\*.tif --dst /path/to/destination/folder/\*.tif

mbnet -lb True -mw 8 -s /path/to/source/folder/*.tif -d /path/to/destination/folder/*.tif

### use package
from mbnet.microbeNet import *

im = pre_processing(im,mean_width = 10)

y = mbnet.segment(im,invert)

y = post_processing(y)
