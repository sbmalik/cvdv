# CVDV: Computer Vision Data Visualization

A data visualization library for computer vision

![Alt_text](/utils/images/cvdv_cover.png)


## Object Detection


![Alt_text](/utils/images/obd_cover.png)

Image Source: [Traffic Signs Dataset in YOLO format](https://www.kaggle.com/valentynsichkar/traffic-signs-dataset-in-yolo-format)

**NOTE**: _OBJECT DETECTION DATA MUST BE YOLO FORMATTED_

CVDV provides you the full analytics of your object detection dataset formatted according to the YOLO algorithm. The analysis performed using `cvdv` can help you in understanding the dataset. It can identify the class imbalance and bounding box size distribution according to different datasets.

---

### 1. Installation

---

    git clone https://github.com/m3sibti/cvdv.git
    cd cvdv
    pip install -r requirements.txt

---

### 1. Running CVDV

    python main.py --data_dir ./path/to/dataset --im_size XX

1.1 **Parameters:**

    --data_dir: Path of dataset directory

    --details_level: Levels of details you wannt to fetch
        . default: only class level information, or leave empty
        . all: for image level and object level information

    --im_size: Size of the images for [SQUARE IMAGES]

    --im_h: Height of the image for [NON-SQUARE IMAGES]
    --im_w: Width of the image for [NON-SQUARE IMAGES]

---

### 2. Visualizations

Followings are the supported types of visualization in `cvdv`. The datset used for this analysis is available on Kaggle, [Traffic Signs](https://www.kaggle.com/valentynsichkar/traffic-signs-dataset-in-yolo-format).

<br/>

2.1 **Class Distribution:**

![ALT Text](/utils/images/class_distribution.png)

<br>

2.2 **Bounding Box Pixel Histograms**

|                  Danger                  |                  Prohibitory                  |
| :--------------------------------------: | :-------------------------------------------: |
| ![](/utils/images/bb_px_dist_danger.png) | ![](/utils/images/bb_px_dist_prohibitory.png) |

<br>

2.3 **Bounding Box's Mean Size**

![](/utils/images/mean_bbpixel_size.png)

<br>

2.4 **Pixel's Color Co-relation**

|                  Danger                   |                  Prohibitory                   |
| :---------------------------------------: | :--------------------------------------------: |
| ![](/utils/images/color_chart_danger.jpg) | ![](/utils/images/color_chart_prohibitory.jpg) |

---

Thank you for interest, Please provide your feedback.
