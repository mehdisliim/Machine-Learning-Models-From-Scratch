class RandomForestClassifier(DecisionTreeClassifier):
    def __init__(self, max_depth=6, min_samples_split=3,  n_estimators=40, max_features_ratio=1, max_samples_ratio=1, imp_measure='gini', verbose=True):
        DecisionTreeClassifier.__init__(self, max_depth=max_depth, min_samples_split=min_samples_split, imp_measure=imp_measure)
        self.n_estimators = n_estimators
        self.max_features_ratio = max_features_ratio
        self.max_samples_ratio = max_samples_ratio
        self.verbose = verbose
        self.estimators = None
        self.selected_feature_per_estimator = None
    
    def select_data(self, X, y, n_samples, n_features):
        columns = []
        while len(columns) < n_features:
            r = np.random.randint(0, X.shape[1], 1)
            if r not in columns:
                columns.append(r)
        rows = np.random.randint(0, n_samples, n_samples)
        columns = np.array(columns).reshape(-1)
        X = X[:, columns]
        X = X[rows, :]
        y = y[rows]
        return X, y, columns
            
    def train(self, X, y):
        
        n_features = round(X.shape[1]* self.max_features_ratio)
        n_samples = round(X.shape[0]* self.max_samples_ratio)
        if self.max_features_ratio < 0 or self.max_features_ratio == 0 or self.max_features_ratio > 1:
            raise ValueError('max_features_ratio must be positif between (0..1]')
        elif self.max_samples_ratio < 0 or self.max_samples_ratio == 0 or self.max_samples_ratio > 1:  
            raise ValueError('max_samples_ratio must be positif between (0..1]')
        elif n_features == 0:
            raise ValueError('Number of selected features per tree == 0 try to augment the value of max_features_ratio')
        elif n_samples == 0:
            raise ValueError('Number of selected samples per tree == 0 try to augment the value of max_samples_ratio')
        
        estimators = []
        selected_features_per_estimator = []
        for i in range(self.n_estimators):
            data_x, data_y, selected_features = self.select_data(X, y, n_samples=n_samples, n_features=n_features)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, imp_measure=self.imp_measure )
            tree.train(data_x, data_y)
            estimators.append(tree)
            selected_features_per_estimator.append(selected_features)
            if self.verbose:
                print('Tree', i+1, 'Done Training!')
        self.estimators, self.selected_features_per_estimator = estimators, selected_features_per_estimator

    def predict(self, X):
        if self.estimators == None:
            raise Exception('You must train the model before making any prediction! use model.train(X, y)')
        else:
            y_pred = np.zeros((X.shape[0], len(self.estimators)))
            for i in range(len(self.estimators)):
                y_pred[:, i] = self.estimators[i].predict(X[:, self.selected_features_per_estimator[i]])
            preds = []            
            for row in y_pred:
                uniques, m_freq = np.unique(row, return_counts=True)
                preds.append(uniques[np.argmax(m_freq)])
            return preds