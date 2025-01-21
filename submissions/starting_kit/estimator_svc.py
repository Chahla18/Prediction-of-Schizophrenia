#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 21:32:17 2025

@author: chahlatarmoun@gmail.com
"""

# import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ROIsFeatureExtractor(BaseEstimator, TransformerMixin):
    """Select only the 284 ROIs features:"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, :284]


def get_estimator():
    """Build your estimator here."""
    estimator = make_pipeline(
        ROIsFeatureExtractor(), 
        StandardScaler(),
        SVC(
        probability=True,
        random_state=42,
        C=0.0038314948719968436,
        kernel='linear',
        gamma='scale' 
        )
    )
    return estimator