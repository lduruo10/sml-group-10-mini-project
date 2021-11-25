class WithoutOutliersClassifier(base.BaseEstimator, base.ClassifierMixin):
    def __init__(self, outlier_detector, classifier):
        self.outlier_detector = outlier_detector
        self.classifier = classifier

    def fit(self, X, y):
        self.outlier_detector_ = base.clone(self.outlier_detector)
        mask = self.outlier_detector_.fit_predict(X, y) == 1
        self.classifier_ = base.clone(self.classifier).fit(X[mask], y[mask])
        return self

    def predict(self, X):
        return self.classifier_.predict(X)
        
grid = {
    'model__classifier__solver' : ['svd'],
    'model__classifier__tol' : [0.0001,0.0002,0.0003],
    'model__classifier__store_covariance' : [True, False]
}

lda = dis_an.LinearDiscriminantAnalysis()

sc = prp.StandardScaler()
    
lof = nei.LocalOutlierFactor()
    
woc = WithoutOutliersClassifier(lof, lda)

pipeline = Pipeline([
         ('preprocess',sc),
        ('model', woc)
    ])

cv = mdl_sl.RepeatedStratifiedKFold(n_splits=6, n_repeats=10, random_state=1)

search = mdl_sl.GridSearchCV(pipeline, grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose = 3)

results = search.fit(x_array_new, y_array)

print(results.best_score_)

print(results.best_params_)