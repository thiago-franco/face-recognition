import numpy as np

class SimilarityClassifier():
    
    def __init__(self):
        self.X_train = []
        self.y_train = []
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def score(self, X_test, y_test):
        correct = 0
        for index, test in enumerate(X_test):
            prediction = self.predict_one(test)
            if y_test[index] == prediction:
                correct += 1
        score = correct/len(X_test)
        return score
    
    def predict_one(self, image):
        min_diff = np.inf
        prediction = 0
        for i, x in enumerate(self.X_train):
            diff = np.linalg.norm(image - x)
            if diff < min_diff:
                min_diff = diff
                prediction = i
        return self.y_train[prediction]
             
    
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            y_pred.append(self.predict_one(x))
        return y_pred
   