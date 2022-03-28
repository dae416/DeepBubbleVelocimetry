# DeepBubbleVelocimetry

This is a project of Multiphase flow & Flow visualization Lab for bubble detection and mask extraction. The purpose of the project is to automatically extract bubble mask of experimental images from various experimental conditions using deep learning model. More information can be found in the paper (Kim & Park, 2021).

The codes are based on Matterport Mask R-CNN implementation, using ResNet-101 as the backbone and applied transfer learning from pre-trained COCO weights.

The output of the model is as follows:

color mask for input image/video
PNG logical mask for each bubble detected
bubble property txt (centroid, area, axes, orientation)
The repository includes:

Source code of Mask R-CNN built on FPN and ResNet101.
Source code to visualize the color mask of the input image/video.
source code to detect and save logical mask and bubble properties.
