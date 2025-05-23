# PyTorch Learning Journey 🚀

**"Learning is not attained by chance, it must be sought for with ardor and attended to with diligence." — Abigail Adams**

A comprehensive collection of PyTorch implementations, experiments, and learning resources as I master deep learning with PyTorch.

## 📌 Table of Contents
- [Repository Structure](#-repository-structure)
- [Key Learning Topics](#-key-learning-topics)
- [Setup Instructions](#-setup-instructions)
- [Notebooks Overview](#-notebooks-overview)
- [Resources](#-resources)
- [Progress Tracking](#-progress-tracking)
- [How to Contribute](#-how-to-contribute)

## 🗂 Repository Structure
```
pytorch-learning/
│
├── basics/ # Fundamental PyTorch operations
│ ├── tensors/ # Tensor operations and autograd
│ └── dataloaders/ # Custom Dataset and DataLoader examples
│
├── nn/ # Neural network implementations
│ ├── mlp/ # Multilayer perceptrons
│ ├── cnn/ # Convolutional neural networks
│ └── rnn/ # Recurrent neural networks
│
├── computer_vision/ # CV applications
│ ├── classification/ # Image classification tasks
│ └── segmentation/ # Image segmentation tasks
│
├── nlp/ # Natural Language Processing
│ ├── text_classification/ # Text classification models
│ └── language_modeling/ # Language modeling approaches
│
├── experiments/ # Experimental code and prototypes
├── utils/ # Helper functions and utilities
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

## 🎯 Key Learning Topics
- **Core Concepts**
  - Tensors & Autograd
  - Dynamic Computation Graphs
  - CUDA Acceleration

- **Neural Network Fundamentals**
  - Custom Layer Implementation
  - Loss Functions & Optimizers
  - Regularization Techniques

- **Advanced Architectures**
  - ResNets, Transformers, GANs
  - Transfer Learning
  - Distributed Training

## ⚙️ Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pytorch-learning.git
   cd pytorch-learning
   ```

2. Create and activate conda environment:
   ```bash
    conda create -n pytorch-env python=3.8
    conda activate pytorch-env
   ```

4. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```
  Or install PyTorch directly:
  ```bash
   conda install pytorch torchvision torchaudio -c pytorch
  ```


## 📓 Notebooks Overview

Notebook	Description	Key Concepts
00_PyTorch_Fundamentals.ipynb	Introduction to PyTorch Tensors	Tensor ops, GPU transfer

## 📚 Resources
Official Documentation
PyTorch Docs
PyTorch Tutorials

**Recommended Books**
"Deep Learning with PyTorch" by Eli Stevens et al.
"Python Deep Learning" by Ivan Vasilev

### Courses
ZTM Pytorch Bootcamp

## 📈 Progress Tracking
1. Tensor Basics
2. Autograd System
3. Distributed Training
4. ONNX Export

## 🤝 How to Contribute
Fork this repository
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
