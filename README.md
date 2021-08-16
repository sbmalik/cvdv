# CVDV: Computer Vision Data Visualization

A data visualization library for computer vision

## Object Detection

**NOTE**: _OBJECT DETECTION DATA MUST BE YOLO FORMATTED_

CVDV provides you the full analytics of your object detection dataset formatted according to the YOLO algorithm. The analysis performed using `cvdv` can help you in understanding the dataset. It can identify the class imbalance and bounding box size distribution according to different datasets.

---

### 1. Installation

---

    git clone https://github.com/m3sibti/cvdv.git
    cd cvdv
    pip install -r requirements.txt

### 1. Running CVDV

---

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

Thank you for interest, Please provide your feedback.
