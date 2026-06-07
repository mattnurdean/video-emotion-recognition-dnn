# Development of Video-Based Emotion Recognition Using Deep Neural Networks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/mattnurdean/video-emotion-recognition-dnn/blob/main/video_emotion_recognition.ipynb)

## 📄 Publication & Citation
This project is based on peer-reviewed research. If you use this repository, find the methodology helpful, or wish to reference the findings in your academic work, please view or cite the official papers here:

* **IEEE Xplore:** [Read Paper on IEEE Xplore](https://ieeexplore.ieee.org/document/9928867)
* **ResearchGate:** [View ResearchGate Profile](https://www.researchgate.net/publication/365105761_Development_of_Video-Based_Emotion_Recognition_System_using_Transfer_Learning)

---

## 📌 Project Overview
Video-based emotion recognition is a highly dynamic field in computer vision with significant potential across human-computer interaction, healthcare, and security systems. This repository contains the complete implementation of my Bachelor's Final Year Project (FYP), focused on designing, tuning, and evaluating deep learning topologies to automate expression analysis. 

The pipeline extracts human faces from raw video sequences and classifies them into **four fundamental emotions**: 
* Disgust
* Happy
* Sad
* Surprise

### Problem Statement
1. Which deep learning neural architectures offer the most viable performance for video-based facial expressions?
2. How accurate is the resulting expression detection when benchmarked against standardized datasets?
3. What is the explicit impact of optimizer selection (SGD, Adam, RMSprop) on discrete model topologies?
4. How does scaling the mini-batch size impact gradient convergence, total training duration, and final prediction metrics?

---

## 📊 Dataset & Hyperparameters
* **Dataset:** Indian Spontaneous Expression Database (ISED) benchmark dataset.
* **Training Sample Volume:** 1,111 frames
* **Validation Sample Volume:** 729 frames
* **Default Setup:** Epochs = 25, Initial Batch Size = 10, Initial Optimizer = SGD.

---

## ⚙️ Pipeline Architecture
The implementation is consolidated into a single, clean pipeline across the following sequential execution steps:

1. **Video Framing:** Leverages OpenCV (`cv2.VideoCapture`) to systematically decode `.avi` video streams down into individual static frames at targeted intervals.
2. **Face Extraction:** Implements an optimized Haar Cascade classifier (`haarcascade_frontalface_default.xml`) to dynamically localize faces, extract the Region of Interest (ROI), and store cropped facial profiles.
3. **LeNet Topology:** Construction of a classic 2D Convolutional configuration scaled to handle $28 \times 28 \times 3$ image dimensions using `tanh` activations and average pooling layers.
4. **AlexNet Topology:** Construction of a deep, high-capacity deep neural network built for $227 \times 227 \times 3$ image sizes utilizing rectified linear units (`relu`), intermediate `BatchNormalization`, `MaxPooling`, and heavy `Dropout` regularization (0.5) to prevent overfitting.
5. **Evaluation Suite:** Evaluates performance via model tracking utilities that generate validation curves (Loss/Accuracy) and extract macro metrics via classification reports and multi-class confusion matrices.

---

## 📈 Key Findings & Insights

* **Architecture Performance:** AlexNet consistently achieved a higher final prediction accuracy compared to LeNet. This performance boost is directly attributed to its deeper layer design and high-capacity architectural parameters, though it introduced an expected tradeoff in overall processing and training duration.
* **Optimizer Sensitivity:** Hyperparameter tuning proved that neural network architectures respond uniquely to different optimization algorithms:
  * **LeNet** reached optimal classification efficiency when paired with the **RMSprop** optimizer, closely followed by Adam (within a 1.00% margin).
  * **AlexNet** achieved its highest performance when matched with standard **SGD**, heavily outperforming both Adam and RMSprop while keeping training times highly efficient.
* **Batch Sizing Impact:** When scaling the mini-batch size from 10 to 100, both models experienced a sharp decrease in validation accuracy along with an increase in total loss. Because a larger batch size results in fewer training updates (steps) per epoch (e.g., dropping to 11 steps per epoch for a dataset of 1,111 samples), the network undergoes significantly fewer weight updates, proving that smaller mini-batches are crucial for maintaining granular model convergence in data-constrained setups.

---

## 🚀 How to Run the Project
Since this pipeline was originally designed to navigate local computational limitations by using cloud infrastructure, the entire project is packaged inside a single Jupyter Notebook optimized for **Google Colab**.

1. Click the **Open In Colab** badge at the top of this repository.
2. Ensure your directory setup in Google Drive mirrors the structural paths defined in the notebook:
   * Haar Cascade path: `/content/drive/MyDrive/Haar Cascade/`
   * ISED Database path: `/content/drive/MyDrive/Database/`
   * Splitting target directories: `/content/drive/MyDrive/Dataset/Train`, `/Validation`, `/Test`
3. Execute the cells sequentially to build the directories, extract the facial frames, fit the networks, and plot performance metrics.
