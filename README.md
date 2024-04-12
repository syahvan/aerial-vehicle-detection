# Aerial Multi-Vehicle Detection

## Overview

This project focuses on the detection of multiple types of vehicles (cars, trucks, and buses) in aerial imagery using the YOLOv8 object detection framework. The objective is to develop a robust system capable of accurately detecting vehicles from aerial views, particularly useful for surveillance, urban planning, and traffic monitoring applications. Additionally, it includes modifications to the YOLOv8 backbone by tweaking the YAML configuration files for experimentation and performance comparison. The dataset used in this project is the Aerial Multi-Vehicle Detection Dataset from Zenodo, which has been preprocessed and cleaned.

## Model Architecture

YOLOv8 is the latest version of the YOLO (You Only Look Once) object detection model, developed by Ultralytics. This model is designed to be an anchor-free model, which means it predicts the center of an object directly instead of using anchor boxes. This reduces the number of box predictions, speeding up the Non-Maximum Suppression (NMS) process, a post-processing step that filters out incorrect predictions.

The YOLOv8 architecture consists of several components:

1. **Backbone**: A series of convolutional layers that extract relevant features from the input image.
2. **SPPF layer**: A layer designed to speed up computation by pooling features of different scales into the network.
3. **Upsample layers**: These layers increase the resolution of the feature maps.
4. **C2f module**: This module combines high-level features with contextual information to improve detection accuracy.
5. **Detection module**: This module uses convolution and linear layers to map high-dimensional features to the output bounding boxes, objectness scores, and class probabilities.

YOLOv8 is known for its speed and efficiency, while still achieving high detection accuracy. It is designed to be flexible and can be used for object detection, image classification, and image segmentation tasks. The model is under active development and receives long-term support from Ultralytics, who work with the community to improve its performance.

<p align="center">
  <img src="https://raw.githubusercontent.com/syahvan/aerial-vehicle-detection/main/asset/Yolov8-Architecture.png" width="85%" height="85%">
  <br>
  Picture 1. YOLOv8 Architecture
</p>

In this project, modifications were made to the backbone architecture of YOLOv8 by altering the YAML configuration files. These modifications aimed to improve detection accuracy or computational efficiency. Specifically, comparisons were made between the standard YOLOv8 backbone and the modified backbone to analyze their impact on detection performance and speed.

<p align="center">
  <img src="https://raw.githubusercontent.com/syahvan/aerial-vehicle-detection/main/asset/Modified-Yolov8-Architecture.png" width="85%" height="85%">
  <br>
  Picture 2. Modified YOLOv8 Architecture
</p>

## Project Steps

1. **Dataset Download**: Start by downloading the preprocessed dataset via this [link](https://zenodo.org/records/7053442). Once downloaded, place it in an appropriate working folder.
2. **Data Preprocessing**: Resize images, augment data (e.g., random cropping, flipping), and label bounding boxes for each vehicle category.
3. **Model Configuration**: Adjust hyperparameters, such as learning rate, batch size, and number of epochs.
4. **Model Training**: Both models are trained in this step. Train the YOLOv8 model on the dataset until convergence, monitoring loss and performance metrics.
5. **Model Evaluation**: Evaluate the trained model using metrics like mean Average Precision (mAP) and Intersection over Union (IoU) on a validation set to assess its performance.
6. **Model Testing and Visualization**: A Jupyter notebook `Run on Video.ipynb` has been created to test the model and visualize detections.

## Performance Comparison

The performance comparison between the standard YOLOv8 backbone and the modified backbone includes metrics such as:

- **Detection Accuracy**: Evaluate mAP, precision, and recall for each vehicle category.
- **Speed**: Measure training time and inference time per image to assess computational efficiency.

Training is carried out using the parameters:

| Device       | Epoch  | Workers | Batch   | Patience |
|--------------|--------|---------|---------|----------|
| GPU Tesla T4 | 100    | 2       | 12      | 50       |

**Note**: This comparison was carried out using the YOLOv8l variant model.

### Detection Accuracy

Table 1. YOLOv8 Evaluation Metrics
| Class    | mAP     | Precision | Recall    |
|----------|---------|-----------|-----------|
| All      | 0.77    | 0.78      | 0.72      |
| Car      | 0.86    | 0.94      | 0.75      |
| Bus      | 0.75    | 0.78      | 0.69      |
| Truck    | 0.68    | 0.63      | 0.72      |

Table 2. Modified YOLOv8 Evaluation Metrics
| Class    | mAP     | Precision | Recall    |
|----------|---------|-----------|-----------|
| All      | 0.82    | 0.83      | 0.75      |
| Car      | 0.92    | 0.95      | 0.81      |
| Bus      | 0.78    | 0.79      | 0.70      |
| Truck    | 0.75    | 0.73      | 0.73      |

It can be seen that the modified YOLOv8 model has better accuracy than the standard YOLOv8 model.

### Speed

| Model           | Training Time (100 Epoch) | Average FPS |
|-----------------|---------------------------|-------------|
| YOLOv8          | 2.6 Hour                  | 29 FPS      |
| Modified YOLOv8 | 2.4 Hour                  | 38 FPS      |

It can be seen that the modified YOLOv8 model is faster than the standard YOLOv8 model.

## Model Usages

Here are some examples of what the model is capable of:

<p align="center">
  <video controls width="85%">
    <source src="https://youtu.be/-GaxSPoVuqE?si=b-1sklTqhUx4WVfj" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <br>
  Video 1. With YOLOv8
</p>

<p align="center">
  <video controls width="85%">
    <source src="https://youtu.be/eHzwjScBaUs?si=7WGDmXp0oHzvBghJ" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <br>
  Video 2. With Modified YOLOv8
</p>

## Reference

Rafael Makrigiorgis, Panayiotis Kolios, & Christos Kyrkou. (2022). Aerial Multi-Vehicle Detection Dataset (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7053442

Ultralytics. (2024). Ultralytics. [Online]. Available: https://github.com/ultralytics/ultralytics


