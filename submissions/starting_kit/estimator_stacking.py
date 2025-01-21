#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 21:32:17 2025

@author: chahlatarmoun@gmail.com
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import NuSVC, SVC
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ROIsFeatureExtractor(BaseEstimator, TransformerMixin):
    """Select only the 284 ROIs features:"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, :284]


def get_estimator():
    """Build your stacking model here."""
    # Base models
    reglog = make_pipeline(
        ROIsFeatureExtractor(),
        StandardScaler(),
        LogisticRegression( 
            C=0.1,
            max_iter=10000,                
            penalty='l1', 
            solver= 'liblinear',
            random_state=42,
        )
    )

    svc_model = make_pipeline(
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

    nusvc_model = make_pipeline(
        ROIsFeatureExtractor(),
        StandardScaler(),
        NuSVC(
            probability=True,
            random_state=42,
            nu=0.6,
            kernel='linear',
            gamma='auto'
        )
    )

    # Stacking model
    estimator = StackingClassifier(
        estimators=[
            ('reglog', reglog),
            ('svc', svc_model),
            ('nusvc', nusvc_model)
        ],
        final_estimator=LogisticRegression(random_state=42),  # Meta-model
        cv=5,
        n_jobs=1
    )

    return estimator