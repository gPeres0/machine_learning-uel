import numpy as np
from collections import defaultdict

class KNN:
    def __init__(self, k=3, metric="euclidean", p=2):
        """
        k: número de vizinhos
        metric: 'euclidean', 'manhattan', 'chebyshev', 'minkowski'
        p: parâmetro da distância de Minkowski
        """
        self.k = k
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _distance(self, x1):
        match self.metric:
            case "euclidean": return [np.sqrt(np.sum((x1 - x2) ** 2)) for x2 in self.X_train]
            case "manhattan": return [np.sum(np.abs(x1 - x2)) for x2 in self.X_train]
            case "chebyshev": return [np.max(np.abs(x1 - x2)) for x2 in self.X_train]
            case "minkowski": return [np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p) for x2 in self.X_train] 
            case _:
                raise ValueError("Métrica desconhecida.")

    def _predict_point(self, x):
        # Calcula todas as distâncias para o ponto x
        distances = self._distance(x)
        
        # Ordena e pega os k vizinhos
        k_indices = np.argsort(distances)[:self.k]
        k_neighbors = self.y_train[k_indices]
        k_distances = np.array(distances)[k_indices]
        
        # Contagem com votos ponderados pela distância
        votes = defaultdict(float)
        for label, dist in zip(k_neighbors, k_distances):
            weight = 1 / (dist + 1e-8)  # evita divisão por zero
            votes[label] += weight
        
        # Classe com maior peso
        return max(votes.items(), key=lambda item: item[1])[0]

    def predict(self, X_test):
        X_test = np.array(X_test)
        return np.array([self._predict_point(x) for x in X_test])
