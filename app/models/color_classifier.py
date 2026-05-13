import importlib
import pickle
from itertools import product
import cv2 as cv
from app.enums.colors import *
import numpy as np


def _patch_sklearn_distance_metric_pickle_aliases():
    """Patch sklearn distance-metric names used by pickled KNN models across versions."""
    dist_metrics = importlib.import_module("sklearn.metrics._dist_metrics")

    # scikit-learn 1.3+ pickles may reference EuclideanDistance64, while older
    # 1.2.x installations expose the same metric as EuclideanDistance.  Adding
    # the alias before pickle.load keeps older environments from failing with:
    #   Can't get attribute 'EuclideanDistance64' on sklearn.metrics._dist_metrics
    if not hasattr(dist_metrics, "EuclideanDistance64") and hasattr(dist_metrics, "EuclideanDistance"):
        setattr(dist_metrics, "EuclideanDistance64", getattr(dist_metrics, "EuclideanDistance"))


class KNNClassifier:
    centers = 1 / 6, 3 / 6, 5 / 6
    patch_size = 8

    def __init__(self, model_path: str) -> None:
        _patch_sklearn_distance_metric_pickle_aliases()
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)


    def _get_color_index(self, image: np.ndarray, center: tuple) -> Color:
        image_height, image_width, _ = image.shape
        center_y, center_x = center

        start_y = int(image_height * center_y - self.patch_size / 2)
        end_y = start_y + self.patch_size

        start_x = int(image_width * center_x - self.patch_size / 2)
        end_x = start_x + self.patch_size

        image_patch = image[start_y:end_y, start_x:end_x]
        patch_color = np.mean(image_patch, axis=(0, 1)).reshape(1, -1)

        return self.model.predict(patch_color)[0]

    def get_colors(self, image: np.ndarray) -> list[Color]:
        if image.size == 0:
            return []

        image = cv.cvtColor(image, cv.COLOR_BGR2LAB)

        return [Color(self._get_color_index(image, (center_y, center_x)))
                for center_y, center_x in product(self.centers, repeat=2)]

    def my_get_colors(self, color_list):
        bgr = np.asarray(color_list, dtype=np.uint8).reshape(-1, 1, 3)
        lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB).reshape(-1, 3)
        return [Color(label) for label in self.model.predict(lab)]




