# MicrobeNet
### Agnostic segmentation of rod-shaped bacteria
Description - pending

## Installation
Requires version python version 3.6

pip install git+https://github.com/pswapnesh/MicrobeNet.git

or 

pip install https://github.com/pswapnesh/MicrobeNet/archive/master.zip


## Usage
### command line
mbnet --light_background True --mean_width 8 --src '/path/to/source/folder/\*.tif' --dst '/path/to/destination/folder/'

mbnet -lb True -mw 8 -s /path/to/source/folder/*.tif -d /path/to/destination/folder/

### use package
from mbnet.microbeNet import *

im = pre_processing(im,mean_width = 10)

y = mbnet.segment(im,invert)

y = post_processing(y)
