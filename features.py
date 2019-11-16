import numpy as np
from numpy.fft import fft
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import pickle


def compute_fft(vector):
    max_points_per_series = 4
    fft_res = np.abs(fft(vector))
    fft_res_list = fft_res.tolist()
    fft_res_list.sort(reverse=True)
    return fft_res_list[:max_points_per_series]


def compute_std(vector):
    return np.std(vector)


class Features:
    def __init__(self, pose_objects):
        self.pose_objects = pose_objects
        self.features = []

    def compute_features(self):
        for pose_object in self.pose_objects:
            feature_vector = []
            # FFT
            feature_vector += compute_fft(pose_object.leftShoulder_x)
            feature_vector += compute_fft(pose_object.leftShoulder_y)
            feature_vector += compute_fft(pose_object.rightShoulder_x)
            feature_vector += compute_fft(pose_object.rightShoulder_y)
            feature_vector += compute_fft(pose_object.leftElbow_x)
            feature_vector += compute_fft(pose_object.leftElbow_y)
            feature_vector += compute_fft(pose_object.rightElbow_x)
            feature_vector += compute_fft(pose_object.rightElbow_y)
            feature_vector += compute_fft(pose_object.leftWrist_x)
            feature_vector += compute_fft(pose_object.leftWrist_y)
            feature_vector += compute_fft(pose_object.rightWrist_x)
            feature_vector += compute_fft(pose_object.rightWrist_y)

            # Variance
            feature_vector += compute_std(pose_object.leftShoulder_x)
            feature_vector += compute_std(pose_object.leftShoulder_y)
            feature_vector += compute_std(pose_object.rightShoulder_x)
            feature_vector += compute_std(pose_object.rightShoulder_y)
            feature_vector += compute_std(pose_object.leftElbow_x)
            feature_vector += compute_std(pose_object.leftElbow_y)
            feature_vector += compute_std(pose_object.rightElbow_x)
            feature_vector += compute_std(pose_object.rightElbow_y)
            feature_vector += compute_std(pose_object.leftWrist_x)
            feature_vector += compute_std(pose_object.leftWrist_y)
            feature_vector += compute_std(pose_object.rightWrist_x)
            feature_vector += compute_std(pose_object.rightWrist_y)

            self.features.append(feature_vector)

    def get_features(self):
        if len(self.features) == 0:
            self.compute_features()
        return self.features

    def compute_pca(self, n_components=20):
        pca = PCA(n_components=n_components, random_state=42)
        scaled_feature_matrix = scale(self.get_features())
        reduced_feature_matrix = pca.fit_transform(scaled_feature_matrix)
        reduced_feature_matrix = reduced_feature_matrix[:, :n_components]
        pickle.dump(pca, open('pca.pkl', 'wb'))
        return reduced_feature_matrix


