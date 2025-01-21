# Prediction-of-Schizophrenia-using-Brain-Anatomy

This project explores the application of machine learning to predict schizophrenia from structural MRI data, aiming to contribute to early detection and better clinical management of this severe mental disorder. Schizophrenia is characterized by significant cognitive, emotional, and behavioral disruptions, often leading to reduced life expectancy. This study utilizes grey matter measurements extracted from MRI scans to develop predictive models, focusing on Regions of Interest (ROIs) for optimal balance between information richness and computational feasibility.

## Project Overview

- **Objective:** Develop a predictive model that differentiates between patients with schizophrenia and healthy controls based on measurements of grey matter volumes from brain MRI scans.

- **Dataset:** There are 410 samples in the training set and 103 samples in the test set.

- **Input data:** Voxel-based_morphometry [VBM](https://en.wikipedia.org/wiki/Voxel-based_morphometry)
using [cat12](http://www.neuro.uni-jena.de/cat/) software which provides:
      
    - Regions Of Interest (`rois`) of Grey Matter (GM) scaled for the Total Intracranial Volume (TIV): 284 features.

    - VBM GM 3D maps or images (`vbm3d`) of [voxels](https://en.wikipedia.org/wiki/Voxel) in the
  [MNI](https://en.wikipedia.org/wiki/Talairach_coordinates) space: contains 3D images of shapes (121, 145, 121). This npz contains the 3D mask and the affine transformation to MNI referential. Masking the brain provide *flat* 331 695 input features (voxels) for each participant.


## Methodology

1. **Preprocessing**:
   - Standardized the data using `StandardScaler` to ensure features had zero mean and unit variance.
   - Focused on **Regions of Interest (ROIs)**, excluding high-dimensional voxel-based data due to computational constraints.

2. **Models Tested**:
   - Logistic Regression with L1 regularization.
   - Support Vector Machine (SVM).
   - Nu-SVC.
   - Ensemble models: Voting Classifier and Stacking.

3. **Final Model**:
   - The **Weighted Voting Classifier** combining Logistic Regression, SVM, and Nu-SVM provided the best performance.
   - **Weights**: Logistic Regression (2.0), SVM (0.01), Nu-SVM (1.5).
   - **Performance**:
     - AUC (Test): **0.8455**
     - Balanced Accuracy (Test): **0.7619**

## Results and Insights

- The Voting Classifier demonstrated the best balance between predictive accuracy and generalization.
- SHAP (SHapley Additive exPlanations) was used to analyze feature importance, identifying key brain regions contributing to the predictions.
- Key predictors include **rPal GM Vol** and **lMedFroCbr CSF Vol**, which were strongly associated with the likelihood of schizophrenia.

## Installation

This starting kit requires Python and the following dependencies:

* `numpy`
* `scipy`
* `pandas`
* `scikit-learn`
* `matplolib`
* `seaborn`
* `jupyter`
* `ramp-workflow`

To run a submission and the notebook you will need the dependencies listed in requirements.txt.
You can install the dependencies with the following command-line:

```
pip install -U -r requirements.txt
```

## Getting started

1. Download the data locally:

```
python download_data.py
```

2. To run locally:

```
ramp-test --submission starting_kit
```