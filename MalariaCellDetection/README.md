# Malaria Cell Detection Project

## DataSet

The MalariaSD dataset encompasses diverse stages and classes of malaria parasites, including Plasmodium falciparum, Plasmodium malariae, Plasmodium vivax, and Plasmodium ovale, categorized into four phases: ring, schizont, trophozoite, and gametocyte. This resource aids researchers and healthcare professionals in gaining critical insights into malaria's epidemiology, diagnosis, and treatment.The MP-IDB, featuring high-quality malaria parasite images with the four stages, offers an opportunity for developing and evaluating novel image processing and analysis techniques to enhance malaria diagnosis accuracy and efficiency. In our proposed paper, we utilized these images to create a new dataset using stable diffusion and advanced image processing methods.Applying stable diffusion, we generated a dataset with 16 distinct classes, focusing on single-celled images. Through cropping and enhancement techniques, we produced refined images. This new dataset underwent training via stable diffusion, resulting in an additional 20 images for each class. Our efforts significantly increased the image count from an average of 12 to 40 images per class in the original dataset. This paper contributes to malaria research by expanding the dataset through stable diffusion and image processing. The augmented dataset provides a more comprehensive representation of malaria parasite stages and classes, empowering researchers and healthcare professionals to enhance their understanding of malaria's complexities and improve diagnostic methodologies. The dataset contains 2 folders

**[Reference](https://ieee-dataport.org/documents/malariasd-malaria-infected-cell-images-dataset)**

- Infected
- Uninfected
- And a total of 27,558 images.
  
You can download the dataset in kaggle:

**[Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)**

## Problem Definition
Malaria is a life-threatening disease caused by parasites transmitted through infected mosquitoes. Rapid and accurate diagnosis is crucial for effective treatment. This project develops a deep learning model to classify cell images as either infected or uninfected with malaria parasites, assisting healthcare professionals in diagnosis.

## Project Structure

```
malaria_project/
    ├── data/ # Dataset (not included in repo)
    ├── notebooks/ 
    ├── models/ 
    ├── output/
    │ ├── OutputImages
    └── requirements.txt
    └── README.md
```


## Technical Approach

### 1. Data Collection and Understanding
- **Dataset**: [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- **Classes**: 
  - Parasitized (infected)
  - Uninfected
- **Image Count**: 27,558 total images
- **Exploration**:
  - Visual inspection of sample images
  - Analysis of image dimensions and distributions

### 2. Data Preprocessing
- Image resizing to 64x64 pixels
- Normalization (pixel values scaled to [0,1])
- Data augmentation:
  - Rotation (±20°)
  - Width/height shifting (±10%)
  - Shearing (±10%)
  - Zooming (±10%)
  - Horizontal flipping
- Train/Validation/Test split (70%/15%/15%)

### 3. Model Architecture Design
```python
Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.2),
    
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.2),
    
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.2),
    
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```
### 4. Training Strategy
Optimizer: Adam (lr=0.001)

Loss: Binary crossentropy

Metrics: Accuracy + AUC

Callbacks:

- Early stopping (patience=5)

- Learning rate reduction on plateau

Epochs: 30

Batch Size: 32

### 5. Model Evaluation
```
Dataset	    Accuracy	AUC
Training	95.31%	    0.988
Validation	92.81%	    0.981
Test	    95.66%	    0.990
```
```
Classification Report (Test Set):

              precision  recall  f1-score  support

Parasitized       0.97     0.94      0.96    13779
Uninfected       0.94      0.97      0.96    13779

accuracy                           0.96     27558
```

### 6. Save the Model for Deployment
```python
#Save model
model.save('malaria_detection_model.keras')
print('Model save successfully!')
```

7. Continuous Improvement
Model Improvements:

Experiment with larger architectures (ResNet, EfficientNet)

Hyperparameter tuning

Data Improvements:

Collect more diverse samples

Add difficult edge cases


Getting Started
Prerequisites
```
Python 3.8+
TensorFlow 2.x
```

Installation
```
bash
git clone https://github.com/francisuml/malaria_project.git
cd malaria_project/api
pip install -r requirements.txt
```


## Author
Francis Carl Sumile
Machine Learning & Deep Learning Enthusiast | Data Science
github/francisuml
