# Microbenet
### Agnostic segmentation of rod-shaped bacteria

## Installation
pip install git++
May need to install tensorflow 2.0 separately.


## Usage
### command line
mbnet --light_background True --mean_width 8 -src /path/to/source/folder/*.tif -dst /path/to/destination/folder/*.tif
mbnet -lb True -mw 8 -s /path/to/source/folder/*.tif -d /path/to/destination/folder/*.tif

### 
from mbnet.microbeNet import *
im = pre_processing(im,mean_width)
y = mbnet.segment(im,invert)
y = post_processing(y)