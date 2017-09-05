# Labeling superpixel colorfulness with Openframeworks

This example code takes in a live video image and creates pixel clusters based on the <a href="http://ivrl.epfl.ch/research/superpixels">SLIC Superpixel Segmentation</a>.

I used this ofxAddon <https://github.com/zrispo/ofxSuperpixels> which is a wrapper for <https://github.com/PSMM/SLIC-Superpixels>.

In addition each pixels cluster is evaluated for it's "colorfulness" by implementing part of this code:
<http://www.pyimagesearch.com/2017/06/26/labeling-superpixel-colorfulness-opencv-python/#comment-428093>

# Dependencies

openframeworks
http://openframeworks.cc/download/

ofxSuperpixels
https://github.com/zrispo/ofxSuperpixels


# Operating systems
It has only been tested on osx 10.10.5 with OF 0.9.8

## Images
Screen shot:

![](https://raw.githubusercontent.com/stephanschulz/superpixels-colorfulness/master/Screen_Shot.png)
