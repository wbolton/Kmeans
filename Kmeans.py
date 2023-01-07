# from Typing import Optional, Union
import itertools
import operator

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import linalg
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler

# scroll down to the bottom to implement your solution


def plot_comparison(
    data: np.ndarray,
    predicted_clusters: np.ndarray,
    true_clusters = None,
    centers = [],
    show: bool = True,
):

    # Use this function to visualize the results on Stage 6.

    if true_clusters is not None:
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        sns.scatterplot(
            x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette="deep"
        )
        if centers is not None:
            sns.scatterplot(
                x=centers[:, 0], y=centers[:, 1], marker="X", color="k", s=200
            )
        plt.grid()

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_clusters, palette="deep")
        if centers is not None:
            sns.scatterplot(
                x=centers[:, 0], y=centers[:, 1], marker="X", color="k", s=200
            )
        plt.grid()
    else:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(
            x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette="deep"
        )
        if centers is not None:
            sns.scatterplot(
                x=centers[:, 0], y=centers[:, 1], marker="X", color="k", s=200
            )
        plt.grid()

    plt.savefig("Visualization.png", bbox_inches="tight")
    if show:
        plt.show()


class CustomKMeans:
    def __init__(self, k: int, max_iter: int = 1000):
        self.k: int = k
        self.centres = []
        self.lp_norm = None  # Optional[Union[int, str]]
        self.iterations: int = 0
        self.max_iter = max_iter

    def fit(self, feature_array: np.ndarray, eps=1e-6, verbose: bool = False) -> None:
        if len(self.centres) == 0:
            self.centres = feature_array[: self.k]
        new_centres = self.calculate_new_centre(feature_array=feature_array)
        self.iterations += 1
        diff = linalg.norm(x=(new_centres - self.centres), ord=self.lp_norm)
        if self.iterations == self.max_iter:
            print(f"Max iterations {str(self.max_iter)} reached")
        elif diff > eps:
            self.centres = new_centres
            self.fit(feature_array=feature_array, verbose=verbose)
        elif verbose:
            print(f"Fitting completed after {str(self.iterations)} iterations")

    def predict(self, feature_array: np.ndarray) -> np.ndarray:
        return self.find_nearest_centre(feature_array=feature_array)

    def predict_coords(self, feature_array: np.ndarray) -> list:
        return self.find_nearest_centre_coords(feature_array=feature_array)

    def find_nearest_centre(self, feature_array: np.ndarray) -> np.ndarray:
        centroids_count = np.shape(self.centres)[0] # type: ignore        
        data_input_count = np.shape(feature_array)[0]
        closest_centroids = np.empty((0, data_input_count), int)
        for row in feature_array:
            distances = np.empty((0, centroids_count), float)
            for centroid in self.centres: # type: ignore
                dist = linalg.norm(x=(row - centroid), ord=self.lp_norm)
                distances = np.append(distances, dist)
            min_dist = distances.argmin()
            closest_centroids = np.append(closest_centroids, min_dist)
        return closest_centroids

    def find_nearest_centre_coords(self, feature_array: np.ndarray) -> list:
        centre_labels = self.find_nearest_centre(feature_array=feature_array)
        coords = [self.centres[x] for x in centre_labels] # type: ignore
        return coords

    def calculate_new_centre(self, feature_array: np.ndarray, lp_norm=None) -> np.ndarray:
        cluster_labels = self.find_nearest_centre(feature_array=feature_array)
        clusters = list(zip(cluster_labels, feature_array))
        means = {}
        # Get unique clusters
        for cluster in set(cluster_labels):
            tmp = [point[1] for point in clusters if point[0] == cluster]
            means[cluster] = np.mean(tmp, axis=0)
        return np.array(list(means.values()))


def find_appropriate_k(feature_array: np.ndarray, p: float = 0.01, max_k: int = 10) -> int:
    # BELOW HERE IS NEW
    errorslist = []
    error_reduction = 1
    sse = None

    for k in range(1, max_k + 1):
        model = CustomKMeans(k=k)
        current_k = k
        model.fit(feature_array=feature_array)
        centres = model.centres

        predicted_clusters = model.predict(feature_array=feature_array)
        prediction_coords = np.array([centres[i] for i in predicted_clusters])

        zipped = list(zip(prediction_coords, feature_array))
        error_sum = 0
        for x in zipped:
            # error = (linalg.norm(x[0] - x[1])**2)/len(x[0])
            centroid = x[0]
            position = x[1]
            diff = (centroid - position) ** 2
            error = ((centroid - position) ** 2).mean(axis=0)
            error_sum += error
        if sse is not None:
            error_reduction = abs(1 - float(error_sum)/float(sse))
        if error_reduction < p:
            break
        sse = error_sum
        errorslist.append(sse)

    print(errorslist)
    return current_k


if __name__ == "__main__":

    # Load data
    data = load_wine(as_frame=True, return_X_y=True)
    X_full, y_full = data

    # Permute it to make things more interesting
    rnd = np.random.RandomState(42)
    permutations = rnd.permutation(len(X_full))
    X_full = X_full.iloc[permutations]  # type: ignore   
    y_full = y_full.iloc[permutations]  # type: ignore   

    # From dataframe to ndarray
    X_full = X_full.values  # type: ignore   
    y_full = y_full.values  # type: ignore   

    # Scale data
    scaler = MinMaxScaler()
    X_full = scaler.fit_transform(X_full)

    find_appropriate_k(feature_array=X_full)
