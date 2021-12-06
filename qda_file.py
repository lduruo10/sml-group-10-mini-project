import numpy as np
import pandas as pd
import sklearn.base as base
import sklearn.covariance as cov
import sklearn.discriminant_analysis as dis_an
import sklearn.metrics as met
import sklearn.model_selection as mdl_sl
import sklearn.preprocessing as prp
from sklearn.pipeline import Pipeline


# QDA follows exactly the LDA procedure so we just included the parts
# that actually produce the final result

# Function to implement the outlier removal process within the pipeline
class WithoutOutliersClassifier(base.BaseEstimator, base.ClassifierMixin):
    def __init__(self, outlier_detector, classifier):
        self.outlier_detector = outlier_detector
        self.classifier = classifier

    def fit(self, X, y):
        self.outlier_detector_ = base.clone(self.outlier_detector)
        mask = self.outlier_detector_.fit_predict(X, y) == 1
        self.classifier_ = base.clone(self.classifier).fit(X[mask], y[mask])
        print(len(X))
        print(len(X[mask]))
        return self

    def predict(self, X):
        return self.classifier_.predict(X)

    def predict_proba(self, X):
        return self.classifier_.predict_proba(X)

    def decision_function(self, X):
        return self.classifier_.decision_function(X)


# Function of final model
def final_model(x_array, y_array):
    qda = dis_an.QuadraticDiscriminantAnalysis(reg_param=0.001, store_covariance=True, tol=0.00001)

    elen = cov.EllipticEnvelope(support_fraction=1.2, store_precision=True, contamination=0.03, assume_centered=True,
                                random_state=1)

    woc = WithoutOutliersClassifier(elen, qda)

    sc = prp.StandardScaler(with_mean=True, with_std=True)

    pipeline = Pipeline([
        ('preprocess', sc),
        ('model', woc)
    ])

    cv = mdl_sl.RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

    scores = mdl_sl.cross_val_score(pipeline, x_array, y_array, scoring='accuracy', cv=cv, n_jobs=-1)

    print("Accuracy: ", np.mean(scores), " Std. Deviation: ", np.std(scores))

    return np.mean(scores), np.std(scores)


# Function for extracting evaluation metrics
def produce_evaluation(x_array, y_array):
    qda = dis_an.QuadraticDiscriminantAnalysis(reg_param=0.001, store_covariance=True, tol=0.00001)

    elen = cov.EllipticEnvelope(support_fraction=1.2, store_precision=True, contamination=0.03, assume_centered=True,
                                random_state=1)

    woc = WithoutOutliersClassifier(elen, qda)

    sc = prp.StandardScaler(with_mean=True, with_std=True)

    pipeline = Pipeline([
        ('preprocess', sc),
        ('model', woc)
    ])

    metrics = {
        'accuracy': 'accuracy',
        'f1': met.make_scorer(met.f1_score, pos_label='Male'),
        'roc_auc': 'roc_auc',
        'precision': met.make_scorer(met.precision_score, pos_label='Male'),
        'recall': met.make_scorer(met.recall_score, pos_label='Male')}

    cv = mdl_sl.RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

    result = mdl_sl.cross_validate(pipeline, x_array, y_array, scoring=metrics, cv=cv, n_jobs=-1)

    labels = []
    avg = []
    std = []

    for m in metrics:
        values = result["test_" + m]
        labels.append(m)
        avg.append(np.mean(values))
        std.append(np.std(values))

    return pd.DataFrame({"Metric": labels, "Average": avg, "Std dev": std})
