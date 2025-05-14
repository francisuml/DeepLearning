# Heart Disease Classification using TensorFlow

## Overview

This repository contains a machine learning project that builds a binary classification model to predict the presence of heart disease using the TensorFlow framework. The model is trained on the processed Cleveland dataset from the UCI Machine Learning Repository, which includes 14 key attributes to determine whether a patient has heart disease (values 1-4) or not (value 0).

## Dataset

The dataset used is the Cleveland Heart Disease Database from the UCI Machine Learning Repository. It contains 76 attributes, but this project utilizes a subset of 14 features. The target variable is binary, where:
0 indicates no heart disease.
1 indicates the presence of heart disease (combining original values 1, 2, 3, and 4).

## Key Features
age: Age of the patient
sex: Sex of the patient (1 = male, 0 = female)
cp: Chest pain type
trestbps: Resting blood pressure
chol: Serum cholesterol
fbs: Fasting blood sugar > 120 mg/dl
restecg: Resting electrocardiographic results
thalach: Maximum heart rate achieved
exang: Exercise-induced angina
oldpeak: ST depression induced by exercise
slope: Slope of the peak exercise ST segment
ca: Number of major vessels colored by fluoroscopy
thal: Thalassemia

Notes

- The dataset contains missing values, which are handled by dropping rows with NaN values.

- Patient names and social security numbers have been replaced with dummy values for privacy.

## Installation

To run this project locally, follow these steps:

1. Clone the Repository

`git clone https://github.com/your-username/heart-disease-classification.git
cd heart-disease-classification`

2. Install DependenciesEnsure you have Python 3.9+ installed. Then, install the required libraries using pip:

`pip install tensorflow pandas numpy matplotlib seaborn scikit-learn`

3. Download the Dataset
Obtain the `processed.cleveland.data` file from the UCI Heart Disease dataset.
Place it in the project directory or update the file path in the notebook accordingly.

## Usage
Open the Jupyter notebook Heart_Disease_Classification_TensorFlow.ipynb.
Run the cells sequentially to:
Load and preprocess the dataset.
Build and train a neural network model with TensorFlow.
Visualize the training history (loss and accuracy).
Analyze the model's performance.
The notebook includes visualizations and insights to monitor model performance and detect overfitting.

## Model Architecture
Input Layer: 
Accepts 13 features after dropping the target variable.
Hidden Layers:
- First layer: 64 neurons with ReLU activation and 20% Dropout.
- Second layer: 32 neurons with ReLU activation.
Output Layer: 1 neuron with sigmoid activation for binary classification.
Optimizer: Adam.
Loss Function: Binary cross-entropy.
Metrics: Accuracy.
Early Stopping: Monitored on validation loss with a patience of 5 epochs.

## Results
- The model achieves a high training accuracy (up to 98%) but shows signs of overfitting, with validation accuracy plateauing around 88-90% and validation loss increasing.
- A confusion matrix and performance plots are generated to evaluate the model.

### Insights
- The increasing validation loss and stable training loss indicate overfitting.
- The model's performance on unseen data is limited, suggesting a need for adjustments.

Recommendations
1. Regularization: Add techniques like L2 regularization or increase Dropout rate.
2. Early Stopping Adjustment: Correct the monitor metric to val_loss (currently set to var_loss).
3. Simpler Model: Reduce the number of neurons or layers.
4. Cross-Validation: Implement k-fold cross-validation for robust evaluation.
5. Hyperparameter Tuning: Adjust learning rate, batch size, or number of epochs.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests for any improvements, such as:

- Enhanced preprocessing techniques.
- Model optimization suggestions.
- Additional visualization or evaluation metrics.

License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

UCI Machine Learning Repository for providing the dataset.
TensorFlow and scikit-learn communities for their robust tools and documentation.

## Author

Francis Carl Sumile

Deep Learning and Machine Learning Enthusiast | Data Science
github/francisuml