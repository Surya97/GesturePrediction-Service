import numpy as np
from numpy.fft import fft
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import pickle


def compute_fft(vector):
    max_points_per_series = 10
    fft_res = np.absolute(fft(vector, max_points_per_series))
    return fft_res.tolist()


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
            left_shoulder = []
            left_shoulder_x_fft = compute_fft(pose_object.leftShoulder_x)
            left_shoulder_y_fft = compute_fft(pose_object.leftShoulder_y)
            feature_vector += left_shoulder_x_fft
            feature_vector += left_shoulder_y_fft
            # for i in range(len(left_shoulder_x_fft)):
            #     left_shoulder.append(left_shoulder_x_fft[i])
            #     left_shoulder.append(left_shoulder_y_fft[i])
            # feature_vector += left_shoulder

            right_shoulder = []
            right_shoulder_x_fft = compute_fft(pose_object.rightShoulder_x)
            right_shoulder_y_fft = compute_fft(pose_object.rightShoulder_y)
            feature_vector += right_shoulder_x_fft
            feature_vector += right_shoulder_y_fft
            # for i in range(len(right_shoulder_x_fft)):
            #     right_shoulder.append(right_shoulder_x_fft[i])
            #     right_shoulder.append(right_shoulder_y_fft[i])
            # feature_vector += right_shoulder

            left_elbow = []
            left_elbow_x_fft = compute_fft(pose_object.leftElbow_x)
            left_elbow_y_fft = compute_fft(pose_object.leftElbow_y)
            feature_vector += left_elbow_x_fft
            feature_vector += left_elbow_y_fft
            # for i in range(len(left_elbow_x_fft)):
            #     left_elbow.append(left_elbow_x_fft[i])
            #     left_elbow.append(left_elbow_y_fft[i])
            # feature_vector += left_elbow

            right_elbow = []
            right_elbow_x_fft = compute_fft(pose_object.rightElbow_x)
            right_elbow_y_fft = compute_fft(pose_object.rightElbow_y)
            feature_vector += right_elbow_x_fft
            feature_vector += right_elbow_y_fft
            # for i in range(len(right_elbow_x_fft)):
            #     right_elbow.append(right_elbow_x_fft[i])
            #     right_elbow.append(right_elbow_y_fft[i])
            # feature_vector += right_elbow

            left_wrist = []
            left_wrist_x_fft = compute_fft(pose_object.leftWrist_x)
            left_wrist_y_fft = compute_fft(pose_object.leftWrist_y)
            feature_vector += left_wrist_x_fft
            feature_vector += left_wrist_y_fft
            # for i in range(len(left_wrist_x_fft)):
            #     left_wrist.append(left_wrist_x_fft[i])
            #     left_wrist.append(left_wrist_y_fft[i])
            # feature_vector += left_wrist

            right_wrist = []
            right_wrist_x_fft = compute_fft(pose_object.rightWrist_x)
            right_wrist_y_fft = compute_fft(pose_object.rightWrist_y)
            feature_vector += right_wrist_x_fft
            feature_vector += right_wrist_y_fft
            # for i in range(len(right_wrist_x_fft)):
            #     right_wrist.append(right_wrist_x_fft[i])
            #     right_wrist.append(right_wrist_y_fft[i])
            # feature_vector += right_wrist


            # Variance
            feature_vector.append(compute_std(pose_object.leftShoulder_x))
            feature_vector.append(compute_std(pose_object.leftShoulder_y))
            feature_vector.append(compute_std(pose_object.rightShoulder_x))
            feature_vector.append(compute_std(pose_object.rightShoulder_y))
            feature_vector.append(compute_std(pose_object.leftElbow_x))
            feature_vector.append(compute_std(pose_object.leftElbow_y))
            feature_vector.append(compute_std(pose_object.rightElbow_x))
            feature_vector.append(compute_std(pose_object.rightElbow_y))
            feature_vector.append(compute_std(pose_object.leftWrist_x))
            feature_vector.append(compute_std(pose_object.leftWrist_y))
            feature_vector.append(compute_std(pose_object.rightWrist_x))
            feature_vector.append(compute_std(pose_object.rightWrist_y))

            # Mean
            feature_vector.append(np.mean(pose_object.leftShoulder_x))
            feature_vector.append(np.mean(pose_object.leftShoulder_y))
            feature_vector.append(np.mean(pose_object.rightShoulder_x))
            feature_vector.append(np.mean(pose_object.rightShoulder_y))
            feature_vector.append(np.mean(pose_object.leftElbow_x))
            feature_vector.append(np.mean(pose_object.leftElbow_y))
            feature_vector.append(np.mean(pose_object.rightElbow_x))
            feature_vector.append(np.mean(pose_object.rightElbow_y))
            feature_vector.append(np.mean(pose_object.leftWrist_x))
            feature_vector.append(np.mean(pose_object.leftWrist_y))
            feature_vector.append(np.mean(pose_object.rightWrist_x))
            feature_vector.append(np.mean(pose_object.rightWrist_y))

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


