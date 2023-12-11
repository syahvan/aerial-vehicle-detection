# Aerial Multi-Vehicle Detection

![Python](https://img.shields.io/badge/Python-3.10.9-blue)

## Description

This project aims to implement a system for the detection and classification of vehicles from aerial images. It uses two nested models, namely Detectron2 for detection and ResNet50 for classification. The dataset used in this project is the Aerial Multi-Vehicle Detection Dataset from Zenodo, which has been preprocessed and cleaned.

## Installation

To install the necessary dependencies, execute the following command:

```bash
pip install -r requirements.txt
```

## Project Steps

1. **Dataset Download**: Start by downloading the preprocessed dataset via this link: [Dataset](https://drive.google.com/drive/folders/11KecJM-7Cu9ILGKEBGuVJcrN5Ap_u6eP?usp=sharing). Once downloaded, place it in an appropriate working folder.

2. **Data Preprocessing**: Two scripts are provided to prepare the data for each of the models:

   - `preprocessing_classification.py`: This script extracts vehicles from the dataset along with their respective class for training the classification model. It will create a `bbox_imgs` folder and a `labels.csv` file.

   - `preprocessing_detectron2.py`: This script transforms vehicle annotations into a unified type and makes them compatible with Detectron2. The data will be stored in a `data` folder, with `train` and `val` subfolders corresponding to training and testing data randomly split with a ratio of 0.8. In each of these folders, two folders will be created, `anns` for annotations and `imgs` for images.

3. **Model Training**: Both models are trained in this step.

   - The classification model is trained using a Jupyter notebook `resnet50_classification.ipynb`. This notebook also contains code for evaluating the model's performance and saving the model weights.

   - The detection model is trained using the script `train_detectron2.py`. The performance and model weights will be saved in the `output` folder.

4. **Model Testing and Visualization**: A Jupyter notebook `detection_classification_test.ipynb` has been created to test the model and visualize detections. This notebook uses the FiftyOne package to evaluate the model on various metrics.

5. **Utility File**: The `util.py` file contains all the classes and functions used throughout the project.

## Project Structure

Here is the final structure of the project:

```bash
├── README.md
├── preprocessing_classification.py           
├── preprocessing_detectron2.py
├── resnet50_classification.ipynb
├── train_detectron2.py
├── detection_classification_test.ipynb
├── util.py
├── detection.pth                             # Final Detection Model
├── classifier.pth                            # Final Classification Model
├── data
│ ├── train
│ │ ├── anns
│ │ └── imgs
│ └── val
│   ├── anns
│   └── imgs
├── bbox_imgs                                 # Extracted Vehicles
├── labels.csv                                # Vehicle Classes
└── output
  └── metrics.json                            # Evaluation Metrics for Detection Model

```
## Model Usages

Here are some examples of what the model is capable of:

![Example 1](Examples/example1.png)
*Example of detection and classification on an aerial image*

![Example 2](Examples/example2.png)
*Another example of detection and classification on an aerial image*

### Model Usages with FiftyOne

![Example 2](Examples/gifexample.gif)

*Example with the use of FiftyOne*

## Reference

Rafael Makrigiorgis, Panayiotis Kolios, & Christos Kyrkou. (2022). Aerial Multi-Vehicle Detection Dataset (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7053442

