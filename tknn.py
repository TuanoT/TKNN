from math import sqrt

class Classifier:
    """
    A simple k-nearest neighbors classifier.
    """


    def __init__(self):
        self.X = []
        self.y = []
        self.k = 5


    def fit(self, X: list, y: list, k: int=5):

        # Set X and y
        if len(X) == len(y):
            self.X = X
            self.y = y
        else:
            raise ValueError("X and y must be the same length")
        
        # Set k
        if k > 0 and k < len(y):
            self.k = k
        else:
            raise ValueError("Something about k is off idk")
        

    def predict(self, pred: list):
        """
        Predict the class of a new data point based on k nearest neighbors.
        :param pred: A list of features for the new data point.
        :return: The predicted class label.
        """
        near_targets = []  # list of (target, dist)
        for x, target in zip(self.X, self.y):
            dist = self.euclidean_distance(x, pred)

            # Add dist into near_targets if it's in k nearest
            if len(near_targets) < self.k:
                near_targets.append((target, dist))
                near_targets.sort(key=lambda pair: pair[1], reverse=True)
            else:
                if dist < near_targets[0][1]:
                    near_targets[0] = (target, dist)
                    near_targets.sort(key=lambda pair: pair[1], reverse=True)

        # Count frequencies of near_targets:
        freq = {}
        for target in near_targets:
            freq = freq.get(target, 0) + 1

        return max(freq, key=freq.get)

   
    def euclidean_distance(A: list, B: list) -> float:
        if len(A) != len(B):
            raise ValueError("X values have different dimensions")
        
        total = 0
        for a, b in zip(A, B):
            total += (a - b) ** 2
        
        return sqrt(total)