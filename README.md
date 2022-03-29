# DeepBuubleVelocimetry for Bubble velocity evaluation

This is a project of [Multiphase flow & Flow visualization Lab](https://mffv.snu.ac.kr/). The purpose of the project is to obtain the bubble velocity field from experimental images using deep learning model. More information can be found in the paper (Choi et al. 2022 (to be published)). 

The codes are based on [PWC_Net](https://github.com/NVlabs/PWC-Net) with the pre-trained weights. 

The output of the model is as follows:

- velocity field (techplot file) 
- velocity field contour (png)

The repository includes:

- Source code of implementation of PWC-Net based on the finetuned weights.
- Source code to generate the velocity field (techplot file) and its contour (png).

![Mask Extraction Example](assets/sample_movie.gif)


## Tested environment
This code was tested on the below environment.

- NVIDIA RTX 2080 ti
- Driver 440.95.01
- CUDA 10.2
- cuDNN 7.6.5
- Python 3.7
- TensorFlow 1.14.0
- Keras 2.2.5 and other common packages listed in `requirements.txt`.


## Preparing the input
Prepare images (JPG or PNG or TIF) '_###' 

- For example, folder_001, folder_002 ... 
- Your `path/to/image` become `.../folder`


## How to test your own bubble image/video
1. Clone this repository
1. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
1. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
1. Download trained weights (mask_rcnn_bubble.h5) from the [link](https://drive.google.com/file/d/1BSi4djQtR0QKYEp-nFGsGi0e6UVEx5ug/view?usp=sharing).

1. Run bubble detection script **in `bubble/` directory** to visualize color mask
   (supports only 3-channel jpg image or video)
    ```bash
    bubble$ python3 bubble.py splash --weights=path/to/mask_rcnn_bubble.h5 --image=path/to/image
    ```
    for the video:
    ```bash
    bubble$ python3 bubble.py splash --weights=path/to/mask_rcnn_bubble.h5 --video=path/to/video
    ```
    
1. Run bubble detection script **in `bubble/` directory** to extract logical mask and bubble properties
   (JPG or PNG or TIF images)
    ```bash
    bubble$ python3 bubble.py detect --weights=path/to/mask_rcnn_bubble.h5 --image=path/to/image --results=/path/to/results --folder_num_start=int --folder_num=number of folders --confidence=0.5 to 0.99
    ```
