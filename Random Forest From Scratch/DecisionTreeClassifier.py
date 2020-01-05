class DecisionTreeClassifier(object):
    def __init__(self, max_depth=6, min_samples_split=3, imp_measure='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.imp_measure = imp_measure
        self.tree = None
    
    def impurity_measure(self, y, imp_measure='gini'):
        nichts, data_points = np.unique(y, return_counts=True) 
        p = data_points / sum(data_points)
        if imp_measure == 'entropy':
            return sum(-p*np.log2(p))
        elif imp_measure == 'gini':
            return 1-sum(np.square(p))
    
    def calculate_best_threshold(self, X, y, impurity_measure='gini'):
        X = np.c_[X]
        all_threshs = self.all_thresholds(X)
        maxi = np.inf
        best_threshold = 0
        its_column = 0
        for key in all_threshs.keys():
            for threshold in all_threshs[key]:
                y_above, y_below = y[X[:, key] >= threshold ], y[X[:, key] < threshold ]
                overall_imp = self.overall_impurity(y_above, y_below, impurity_measure)
                if overall_imp <= maxi:
                    maxi = overall_imp
                    best_threshold = threshold
                    its_column = key
        return best_threshold, its_column
    
    def overall_impurity(self, y1, y2, imp_measure='gini'):
        len_y1,  len_y2 = len(y1), len(y2)
        len_data = len_y1+len_y2
        if imp_measure == 'entropy':
            return (len_y1/len_data)*self.impurity_measure(y1, 'entropy')+(len_y2/len_data)*self.impurity_measure(y2, 'entropy')
        elif imp_measure == 'gini':
            return (len_y1/len_data)*self.impurity_measure(y1, 'gini')+(len_y2/len_data)*self.impurity_measure(y2, 'gini')
        else:
            raise NameError('{} is not supported as impurity measure please use gini or entropy'.format(imp_measure))
    def most_frequent(self, y):
        unique_vals, counts_unique_vals = np.unique(y, return_counts=True) #in this case since we told it to return the counts meaning how many each unique value appeared in the array so it will return the unique values and their counts
        return unique_vals[np.argmax(counts_unique_vals)]
    
    def all_thresholds(self, X):
        D = {}
        for c in range(X.shape[1]):
            D[c] = []
            for l in range(X.shape[0]):
                if l > 0:
                    D[c].append((X[l-1, c] + X[l, c])/2)
        return D
        
    def ispure(self, y):
        unique_vals = np.unique(y)
        return len(unique_vals) == 1 

    def create_tree(self, X, y, counts=1, max_depth=-1, min_samples_split=-1):
        max_depth, min_samples_split=self.max_depth , self.min_samples_split
        best_threshold, its_column = self.calculate_best_threshold(X, y, impurity_measure=self.imp_measure)
        X_below, y_below, X_above, y_above = self.split_data_using_threshold(X, y, best_threshold, its_column)
        if len(y) >= min_samples_split and counts <= max_depth and len(y_above) != 0 and len(y_below) != 0:
            counts +=1
            if self.ispure(y_below) == False:
                yes = self.create_tree(X_below, y_below, counts=counts, max_depth=max_depth, min_samples_split=min_samples_split)
            else:
                yes = self.most_frequent(y_below)
            if self.ispure(y_above) == False:
                no = self.create_tree(X_above, y_above, counts=counts, max_depth=max_depth, min_samples_split=min_samples_split )
            else:
                no = self.most_frequent(y_above)
            return {str(its_column)+'_'+str(best_threshold) : {True:yes, False:no}}
        else:
            return {str(its_column)+'_'+str(best_threshold) : {True:self.most_frequent(y), False:self.most_frequent(y)}}
   
    def predict_sample(self, D, X):
        if 'int' in str(type(D)) or 'float' in str(type(D)):
            return D
        else:
            col, thresh = self.col_threshold(tuple(D.keys())[0])
            D = D[tuple(D.keys())[0]]
            D = D[X[col] < thresh]
            return self.predict_sample(D, X)
    
    def predict(self, X):
        X = np.c_[X] #making sure even if it's one sample it gets in the form of 2D
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.predict_sample(self.tree, X[i]))
        return y_pred

    
    def train(self, X, y): #make sure X is a 2D array even if they were DataFrames or Series they r converted to 2D so the code workds
        X, y = np.c_[X], np.c_[y]
        self.tree = self.create_tree(X, y)
    
    col_threshold = lambda self, a: (int(a[:a.index('_')]), float(a[a.index('_')+1:]))
    split_data_using_threshold = lambda self, X, y, threshold, col : (X[X[:, col] < threshold], y[X[:, col] < threshold], X[X[:, col] >= threshold], y[X[:, col] >= threshold] )