# Computer-Vision-projects
some projects implementing basic CV stuff including Harris corner detector, Hough transform for line detection, Histogram of oriented gradients, as well as every ML engineer wannabe's first DL project: image classification using CNNs (classics are classics for a reason)

# Harris Corner Detector
The **Harris corner detector** is commonly used to detect corners of an image. The idea is to locate points of interest where the surrounding neighborhood shows edges in more than one direction. A corner can be detected by looking at large variations in intensities within a small window when moving around. The change can be estimated using the following equation:

$$ E(u, v) = \sum_{x, y} w(x, y) [I(x + u, y + v) - I(x, y)]^2 $$


Where:  
- \( E \) is the difference between the original and the moved window,  
- \( u \) and \( v \) are the window’s displacement in x and y directions,  
- \( w(x, y) \) is the window at position \( (x, y) \),  
- \( I(x+u, y+v) \) is the intensity of the moved window,  
- \( I(x, y) \) is the intensity of the original image at position \( (x, y) \).  

The window function can be a rectangular window or a Gaussian window which gives weights to pixel \( (x, y) \). 

The above equation can be further approximated using Taylor expansion, giving the final formula as:

$$ E(u, v) \approx \begin{bmatrix} u \\ v \end{bmatrix}^T M \begin{bmatrix} u \\ v \end{bmatrix} $$

Where:  

$$ M = \sum_{x, y} w(x, y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix} $$


### Corner Detection Process

1. **Color image to grayscale image conversion**  
2. **Spatial derivatives** (x and y directions → \( G_x \) and \( G_y \))  
3. **Compute products of derivatives** at every pixel (apply smoothing kernel if required)  
4. **Compute the sums of the products of derivatives** at each pixel  
5. Define the matrix \( M \) at each pixel:  
$$
M = \begin{bmatrix} S_x^2 & S_{xy} \\ S_{xy} & S_y^2 \end{bmatrix}
$$
6. Compute the **response of the detector** at each pixel:  
$$
R = \text{det}(M) - k (\text{trace}(M))^2
$$

   Where:  
   - $$\( \text{det}(M) = S_x^2 S_y^2 - S_{xy}^2 \)  $$
   - $$\( \text{trace}(M) = S_x + S_y \)  $$
   - \( k \) is the sensitivity factor to separate corners from edges, typically a value close to zero (usually between 0.04 – 0.06).  
7. **Threshold on value of \( R \)**; compute non-max suppression. All windows with \( R \) greater than a certain value are corners. If \( R \) is a large negative number, it is likely an edge; otherwise, for flat regions, \( R = 0 \).

---
# Hough Transform Implementation

### 1. **Hough Transform Function**
- **Inputs:**
  - `Im`: Edge magnitude image (grayscale).
  - `rhoRes`: Resolution of the accumulator along the $$\( \rho \)$$ axis.
  - `thetaRes`: Resolution of the accumulator along the $$\( \theta \)$$ axis.
- **Outputs:**
  - `H`: Hough transform accumulator matrix containing votes for possible lines.
  - `rhoScale`: Array of $$\( \rho \)$$ values.
  - `thetaScale`: Array of $$\( \theta \)$$ values.

Each edge pixel \( (x, y) \) votes for all possible lines passing through it in the parameter space \( (\rho, \theta) \), where:

$$\[
\rho = x \cos \theta + y \sin \theta
\]$$

### 2. **Detecting Lines**
- Use the accumulator matrix \( H \) to find the top \( nLines \) strongest lines.
- Apply **non-maximal suppression** to remove neighboring peaks in the accumulator to ensure distinct lines are detected.
- **Outputs:**
  - `rhos`: Array of $$\( \rho \)$$ values for detected lines.
  - `thetas`: Array of $$\( \theta \)$$ values for detected lines.

### 3. **Line Segment Fitting**
- For visualization, fit detected lines to segments corresponding to actual edges in the image.
- **Function Output:**
  - `lines`: Array of structures containing:
    - `lines[i].start`: Start point \( (x, y) \) of the segment.
    - `lines[i].stop`: End point \( (x, y) \) of the segment.

### 4. **Visualization**
- Plot the detected line segments on the original image for validation.
- Display intermediate results, including the Hough transform accumulator and detected lines.
- 

---
# Human Detection Using HOG

This notebook implements a system for human detection in 2D color images using the **Histograms of Oriented Gradients (HOG)** feature and a **Two-Layer Perceptron (neural network)** (implemented from scratch to keep things spicy).

---

## Steps

### 1. **Preprocessing**
- Convert the input color image to grayscale using the formula:

$$
I = 0.299R + 0.587G + 0.114B
$$

- Compute horizontal ($$G_x$$) and vertical ($$G_y$$) gradients using the Prewitt operator.
- Calculate:
  - **Edge Magnitude**:

$$
M(i, j) = \sqrt{G_x^2 + G_y^2}
$$

  - **Gradient Angle**:

$$
\theta = \arctan\left(\frac{G_y}{G_x}\right)
$$

    - Gradient angles are measured with respect to the positive x-axis.
    - Assign $$M(i, j) = 0$$ and $$\theta = 0$$ where $$G_x = G_y = 0$$.


### 2. **HOG Feature Extraction**
- **Gradient Angle Quantization**:
  - Divide angles into 9 bins (unsigned representation).
  - Normalize angles between $$[0, 180)$$.
- Parameters:
  - **Cell Size**: $$8 \times 8$$ pixels.
  - **Block Size**: $$16 \times 16$$ pixels (or $$2 \times 2$$ cells).
  - **Block Step**: $$8$$ pixels (1 cell overlap).
- Use **L2 norm** for block normalization. Keep final descriptor values as floating-point numbers.



### 3. **Neural Network for Classification**
- Design a **Two-Layer Perceptron**:
  - **Input Layer**: Size $$N$$ (HOG descriptor size).
  - **Hidden Layer**: Test sizes of $$250$$, $$500$$, and $$1000$$ neurons.
  - **Output Layer**: 1 neuron for binary classification (human/no human).
- Activation Functions:
  - **Hidden Layer**: ReLU.
  - **Output Layer**: Sigmoid (to output probabilities).
- Use the **backpropagation algorithm** to train the network.



## Implementation Notes
- No external libraries for HOG computation or neural network training.
- Use libraries for basic operations like image I/O, matrix arithmetic, and common math functions.
- Train on 20 images:
  - **10 Positive**: Containing humans.
  - **10 Negative**: Without humans.
- Test on 10 images:
  - **5 Positive** and **5 Negative**.

---
# CIFAR-10 Image Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset using PyTorch.

## Project Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes (e.g., airplane, car, bird, etc.). This project includes:
- Loading and normalizing the dataset.
- Implementing a custom PyTorch `Dataset` class and `DataLoader` for batch processing.
- Building a CNN for multi-class classification.
- Training and testing the model using Cross-Entropy Loss and the Adam optimizer.

## Features

- **Custom Dataset Loader**:
  - A utility to parse and preprocess CIFAR-10 batches.
  - Normalizes pixel values to the range [-1, 1].
- **CNN Architecture**:
  - Two convolutional layers with ReLU activation and max-pooling.
  - A fully connected layer for classification.
- **Training Pipeline**:
  - Adam optimizer with a learning rate of `0.0005` and weight decay of `0.001`.
  - Cross-Entropy Loss for classification.


## File Structure

- `image_classification.ipynb`: Main notebook containing the implementation.
- `cifar-10-batches-py/`: Folder containing CIFAR-10 data batches (download required).
- `README.md`: This file, serving as project documentation.

## Getting Started

### 1. Dataset Preparation

Download the CIFAR-10 dataset from the [CIFAR-10 Website](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz). Extract the files into a folder named `cifar-10-batches-py/`.

### 2. Running the Code

Open the Jupyter Notebook and run the cells in sequence. Key steps include:
1. Loading and normalizing the data.
2. Creating training and testing data loaders.
3. Defining the CNN model.
4. Training the model on the CIFAR-10 dataset.
5. Evaluating the model's accuracy on the test set.

### 3. Output

The notebook will print the following:
- Device information (CPU or GPU).
- Model accuracy on the test dataset.

## Model Architecture

| Layer Type    | Parameters                     | Output Shape |
|---------------|--------------------------------|--------------|
| Conv2D        | 3 input channels, 6 filters    | 6x28x28      |
| ReLU          | Activation                     | 6x28x28      |
| MaxPooling    | Kernel size 2x2                | 6x14x14      |
| Conv2D        | 6 input channels, 16 filters   | 16x10x10     |
| ReLU          | Activation                     | 16x10x10     |
| MaxPooling    | Kernel size 2x2                | 16x5x5       |
| FullyConnected| 400 input, 10 output neurons   | 10           |



