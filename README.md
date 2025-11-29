# 🩺 Chest X-Ray Pneumonia Detector

## 🚀 Project Overview

This project develops a **Deep Learning tool** designed to assist medical professionals by automatically flagging chest X-ray images that are highly suggestive of **pneumonia**. Leveraging a fine-tuned **ResNet50** Convolutional Neural Network (CNN), the model acts as a **'second reader'**, highlighting X-rays that warrant immediate, closer inspection.

The goal is to provide a reliable, high-speed automated preliminary assessment to minimize false negatives (missed pneumonia cases) while maintaining a clinically acceptable low false alarm rate.

---

## ✨ Key Features & Methodology

* **Transfer Learning:** Utilizes a pre-trained **ResNet50** model, fine-tuning its final layers on the chest X-ray dataset for rapid convergence and high performance.
* **Data Balancing:** Employed **undersampling** of the majority class (Pneumonia) and a custom **balanced data generator** with **data augmentation** to mitigate class imbalance and prevent overfitting, which is critical for medical datasets.
* **Optimal Threshold Tuning:** The decision threshold was calibrated using **Youden's J statistic** on the validation set to achieve the best balance between **Sensitivity (Recall)** and **Specificity**, maximizing overall diagnostic effectiveness.
* **Evaluation Metrics:** Performance is measured using key clinical metrics like **Accuracy**, **AUC**, **Recall (Sensitivity)**, and the **Confusion Matrix** to ensure robust performance on both Normal and Pneumonia cases.

---

## 💻 Technical Setup & Installation

To run this notebook and train/test the model locally, follow these steps.

### Prerequisites

You need a Python environment with the following major libraries:

* Python 3.x
* **TensorFlow / Keras**
* **OpenCV (`cv2`)**
* **scikit-learn**
* **NumPy**
* **Pandas**

### Data Source

This project requires the **Chest X-Ray Images (Pneumonia)** dataset, which must be organized in the following structure relative to the notebook:

chest_xray/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/


### Run the Project

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/Chest-X-Ray-Pneumonia-Detector.git](https://github.com/yourusername/Chest-X-Ray-Pneumonia-Detector.git)
    cd Chest-X-Ray-Pneumonia-Detector
    ```
2.  Install dependencies:
    ```bash
    pip install tensorflow keras opencv-python scikit-learn numpy pandas
    ```
3.  Place the dataset into the required `./chest_xray/` folder.
4.  Execute the Jupyter Notebook (`.ipynb` file).

---

## 📈 Model Performance & Results

The model was evaluated on an independent test set using the optimally tuned sigmoid threshold of **0.523**.

### Key Test Metrics

| Metric | Value |
| :--- | :--- |
| **Test Loss** | $0.3065$ |
| **Accuracy** | $0.8798$ |
| **AUC (Area Under Curve)** | $0.9434$ |

### Classification Report

| Class | Precision | Recall (Sensitivity) | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Normal (True Negative)** | $0.86$ | $0.81$ | $0.83$ | $234$ |
| **Pneumonia (True Positive)** | $0.89$ | **$0.92$** | $0.91$ | $390$ |

### Confusion Matrix (with Tuned Threshold)

| | **Normal (pred)** | **Pneumonia (pred)** |
| :--- | :--- | :--- |
| **Normal (true)** | $189$ (True Negatives) | $45$ (False Positives) |
| **Pneumonia (true)** | $30$ (False Negatives) | **$360$** (True Positives) |

### **Interpretation**

The model demonstrates a strong ability to detect Pneumonia cases, achieving a **Recall of 92%**. This is a critical result in a clinical setting, as it means the model is excellent at minimizing **False Negatives** (missing an actual case of pneumonia), which is the primary goal of a screening tool. The overall accuracy of approximately **88%** confirms its robust performance across both classes.

---

## 🛠 Model Architecture

The model uses the **ResNet50** architecture as a feature extractor, followed by custom dense layers for classification:

1.  **ResNet50 Base Model** (Frozen weights from ImageNet)
2.  `GlobalAveragePooling`
3.  `Dense` Layer (128 units, ReLU activation)
4.  `Dropout` Layer (0.5)
5.  `Dense` Output Layer (1 unit, Sigmoid activation)

The model was compiled with the `adam` optimizer and `binary_crossentropy` loss.
