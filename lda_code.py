import numpy as np
import pandas as pd
import scipy as sc
import sklearn.base as base
import sklearn.covariance as cov
import sklearn.discriminant_analysis as dis_an
import sklearn.ensemble as ens
import sklearn.feature_selection as feat_sel
import sklearn.metrics as met
import sklearn.model_selection as mdl_sl
import sklearn.neighbors as nei
import sklearn.preprocessing as prp
import sklearn.svm as svm
import sklearn.tree as tree
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Read training dataset
data_path = 'data/train.csv'
data = pd.read_csv(data_path)

# Transform the training dataset data in the desired form
x = data[list(data.columns[:-1])]
x_array = np.array(x)
x.index = list(x.index)
y = data[['Lead']]
y_array = np.array(y)
y_array = y_array.ravel()
y.index = list(y.index)


def apply_lda(x_array, y_array, solver_inp='svd', shrinkage_inp=None, tol_inp=0.0001, covariance_estimator_inp=None):
    lda = dis_an.LinearDiscriminantAnalysis(solver=solver_inp, shrinkage=shrinkage_inp, tol=tol_inp,
                                            covariance_estimator=covariance_estimator_inp)
    cv = mdl_sl.RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

    scores = mdl_sl.cross_val_score(lda, x_array, y_array, scoring='accuracy', cv=cv, n_jobs=-1)

    print("Accuracy: ", np.mean(scores), " Std. Deviation: ", np.std(scores))

    return np.mean(scores), np.std(scores)


# Function to transform data into gaussian distribution
def transform_to_gaussian(x):
    x_temp = x.copy()
    for col in x.columns:
        if col == 'Year':
            x_temp[col] = np.log(x[col])
        else:
            x_temp[col] = sc.stats.boxcox(x[col])[0]

    return x_temp, np.array(x_temp)


# Create a 'dirty' dataset to avoid mistakes in cases where
# zero is observed
x_dirty = x.copy()
x_dirty['Number words female'] += 1
x_dirty['Number words male'] += 1
x_dirty['Gross'] += 1
x_new = x_dirty

# Get a normalized dataset
x_new, x_array_new = transform_to_gaussian(x_new)


# Function which returns the accuracy and standard deviation
# of GridSearchCV
def get_accuracy_std_dev(results):
    return results.best_score_, results.cv_results_['std_test_score'][results.best_index_]


# Function which implements the pipeline with scaling and model
def get_best_parameters_cv_and_scaling_included(x_array, y_array, grid_dict={}):
    lda = dis_an.LinearDiscriminantAnalysis()

    sc = prp.StandardScaler(with_mean=True, with_std=True)

    pipeline = Pipeline([
        ('preprocess', sc),
        ('model', lda)
    ])

    cv = mdl_sl.RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

    search = mdl_sl.GridSearchCV(pipeline, grid_dict, scoring='accuracy', cv=cv, n_jobs=-1, verbose=3)

    results = search.fit(x_array, y_array)

    print(results.best_score_)

    print(results.best_params_)

    return results


# Find the best parameters of LDA
grid = {
    'model__solver': ['svd', 'lsqr', 'eigen'],
    'model__tol': [0.00001, 0.0001, 0.0002, 0.0005, 0.001],
    'model__store_covariance': [True, False],
    'model__shrinkage': list(np.linspace(0, 1, 50))
}

results = get_best_parameters_cv_and_scaling_included(x_array_new, y_array, grid)


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


# Function to decide which outlier removal method to use
def get_best_parameters_cv_scaling_outliers_included(x_array, y_array, grid, outlier_removal_method, random=1):
    lda = dis_an.LinearDiscriminantAnalysis()

    woc = WithoutOutliersClassifier(outlier_removal_method, lda)

    sc = prp.StandardScaler(with_mean=True, with_std=True)

    pipeline = Pipeline([
        ('preprocess', sc),
        ('model', woc)
    ])

    cv = mdl_sl.RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

    if random == 1:
        search = mdl_sl.RandomizedSearchCV(pipeline, grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=3,
                                           random_state=1)
    else:
        search = mdl_sl.GridSearchCV(pipeline, grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=3)

    results = search.fit(x_array, y_array)

    print(results.best_score_)

    print(results.best_params_)

    return results


outliers = [ens.IsolationForest(), cov.EllipticEnvelope(), nei.LocalOutlierFactor(), svm.OneClassSVM()]
outliers_names = ['IsolationForest', 'EllipticEnvelope', 'LocalOutlierFactor', 'OneClassSVM']

grid_lda = {
    'model__classifier__solver': ['lsqr'],
    'model__classifier__tol': [0.00001],
    'model__classifier__store_covariance': [True],
    'model__classifier__shrinkage': [0.02040816326530612]
}

grid_isolation_forest = {
    'model__outlier_detector__n_estimators': list(range(100, 800, 5)),
    'model__outlier_detector__max_samples': list(range(100, 500, 5)),
    'model__outlier_detector__contamination': [0.1, 0.2, 0.3, 0.4, 0.5],
    'model__outlier_detector__max_features': [5, 10, 15],
    'model__outlier_detector__bootstrap': [True, False],
    'model__outlier_detector__n_jobs': [-1]
}

grid_elliptic_envelope = {
    'model__outlier_detector__store_precision': [True, False],
    'model__outlier_detector__assume_centered': [True, False],
    'model__outlier_detector__support_fraction': list(np.linspace(0, 1, 10)),
    'model__outlier_detector__contamination': list(np.linspace(0.01, 0.5, 10)),
    'model__outlier_detector__random_state': [1]
}

grid_local_outlier_factor = {
    'model__outlier_detector__n_neighbors': list(np.linspace(1, 500, 50, dtype='int32')),
    'model__outlier_detector__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'model__outlier_detector__leaf_size': list(np.linspace(10, 100, 10, dtype='int32')),
    'model__outlier_detector__metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'],
    'model__outlier_detector__contamination': list(np.linspace(0.01, 0.5, 10)),
    'model__outlier_detector__novelty': [True, False],
    'model__outlier_detector__n_jobs': [-1]
}

grid_local_one_class_svm = {}

outliers_grid = [grid_isolation_forest, grid_elliptic_envelope, grid_local_outlier_factor, grid_local_one_class_svm]

results_iter = []

for outlier_index in range(0, len(outliers)):
    grid_lda_with_outlier = {**grid_lda, **outliers_grid[outlier_index]}

    print(outliers[outlier_index])

    results = get_best_parameters_cv_scaling_outliers_included(x_array_new, y_array, grid_lda_with_outlier, outliers[outlier_index])

    results_iter.append(results)

    print()


# Function to check feature selection models
def get_best_parameters_feature_selection_cv_scaling_outliers_included(x_array, y_array, grid, feature_selection_method):
    lda = dis_an.LinearDiscriminantAnalysis(solver='lsqr', tol=0.00001, store_covariance=True,
                                            shrinkage=0.02040816326530612)

    lof = nei.LocalOutlierFactor(novelty=False, n_neighbors=250, n_jobs=-1, metric='euclidean', leaf_size=40,
                                 contamination=0.05000000000000001, algorithm='kd_tree')

    woc = WithoutOutliersClassifier(lof, lda)

    sc = prp.StandardScaler(with_mean=True, with_std=True)

    pipeline = Pipeline([
        ('preprocess', sc),
        ('selector', feature_selection_method),
        ('model', woc)
    ])

    cv = mdl_sl.RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

    search = mdl_sl.GridSearchCV(pipeline, grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=3)

    results = search.fit(x_array, y_array)

    print(results.best_score_)

    print(results.best_params_)

    return results


# Select KBest - Anova
sel = feat_sel.SelectKBest(score_func=feat_sel.f_classif)

grid = {
    'selector__k': list(range(1, len(x_new.columns) + 1))
}

results_anova = get_best_parameters_feature_selection_cv_scaling_outliers_included(x_array_new, y_array, grid, sel)

# Wrapper Methods (RFE)
sel = feat_sel.RFE(estimator=tree.DecisionTreeClassifier(), step=1)

grid = {
    'selector__n_features_to_select': list(range(1, len(x_new.columns) + 1))
}

results_rfe = get_best_parameters_feature_selection_cv_scaling_outliers_included(x_array_new, y_array, grid, sel)

# Sequential Feature Selector

grid = {
    'selector__n_features_to_select': list(range(1, len(x_new.columns) + 1))
}

sel = feat_sel.SequentialFeatureSelector(tree.DecisionTreeClassifier())

results_sfs = get_best_parameters_feature_selection_cv_scaling_outliers_included(x_array_new, y_array, grid, sel)


# Function of final model
def final_model(x_array, y_array):
    lda = dis_an.LinearDiscriminantAnalysis(solver='lsqr', tol=0.00001, store_covariance=True,
                                            shrinkage=0.02040816326530612)

    lof = nei.LocalOutlierFactor(novelty=False, n_neighbors=250, n_jobs=-1, metric='euclidean', leaf_size=40,
                                 contamination=0.05000000000000001, algorithm='kd_tree')

    woc = WithoutOutliersClassifier(lof, lda)

    sc = prp.StandardScaler(with_mean=True, with_std=True)

    pipeline = Pipeline([
        ('preprocess', sc),
        ('model', woc)
    ])

    cv = mdl_sl.RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

    scores = mdl_sl.cross_val_score(pipeline, x_array, y_array, scoring='accuracy', cv=cv, n_jobs=-1)

    print("Accuracy: ", np.mean(scores), " Std. Deviation: ", np.std(scores))

    return np.mean(scores), np.std(scores)


# Feature Creation

x_with_new_features = x.copy()
x_with_new_features['words per female actor'] = x_with_new_features['Number words female'] / x_with_new_features[
    'Number of female actors']
x_with_new_features['words per male actor'] = x_with_new_features['Number words male'] / x_with_new_features[
    'Number of male actors']
x_with_new_features['female words vs total words'] = x_with_new_features['Number words female'] / x_with_new_features[
    'Total words']
x_with_new_features['male words vs total words'] = x_with_new_features['Number words male'] / x_with_new_features[
    'Total words']
x_with_new_features['number of words co-lead'] = x_with_new_features['Number of words lead'] - x_with_new_features[
    'Difference in words lead and co-lead']
x_with_new_features['female vs male ratio'] = x_with_new_features['Number of female actors'] / x_with_new_features[
    'Number of male actors']

# Add 1 to zero columns
x_with_new_features['Number words female'] += 1
x_with_new_features['Number words male'] += 1
x_with_new_features['Gross'] += 1
x_with_new_features['words per female actor'] += 1
x_with_new_features['words per male actor'] += 1
x_with_new_features['female words vs total words'] += 1
x_with_new_features['male words vs total words'] += 1

x_with_new_features_new, x_array_with_new_features_new = transform_to_gaussian(x_with_new_features)


# Function for checking collinearity
def check_collinearity(x):
    vif_data = pd.DataFrame()
    vif_data["feature"] = x.columns
    vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
    return vif_data.sort_values(['VIF'], ascending=False)


# Function for extracting evaluation metrics
def produce_evaluation(x_array, y_array):
    lda = dis_an.LinearDiscriminantAnalysis(solver='lsqr', tol=0.00001, store_covariance=True,
                                            shrinkage=0.02040816326530612)

    lof = nei.LocalOutlierFactor(novelty=False, n_neighbors=250, n_jobs=-1, metric='euclidean', leaf_size=40,
                                 contamination=0.05000000000000001, algorithm='kd_tree')

    woc = WithoutOutliersClassifier(lof, lda)

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

    result = mdl_sl.cross_validate(pipeline, x_array, y_array, scoring=metrics, cv=cv, n_jobs=-1, return_estimator=True)

    labels = []
    avg = []
    std = []

    for m in metrics:
        values = result["test_" + m]
        labels.append(m)
        avg.append(np.mean(values))
        std.append(np.std(values))

    return pd.DataFrame({"Metric": labels, "Average": avg, "Std dev": std})
