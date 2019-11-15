from train_preprocessing import Preprocess
from numpy.fft import fft
import numpy as np
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


def compute_fft_features(label_vector):
    feature_vector = []
    max_points_per_series = 4
    cgm_fft = np.abs(fft(label_vector))
    cgm_fft = cgm_fft.tolist()
    cgm_fft.sort(reverse=True)
    feature_vector += cgm_fft[:max_points_per_series]
    return feature_vector

def compute_variance(label_vector):
    return

preprocess = Preprocess()
preprocess.scale_points()


video_objects =preprocess.new_pose_objects


features = []
for video_obj in video_objects:
    feature_vector = []
    #FFT
    feature_vector += compute_fft_features(video_obj.leftShoulder_x)
    feature_vector += compute_fft_features(video_obj.leftShoulder_y)
    feature_vector += compute_fft_features(video_obj.rightShoulder_x)
    feature_vector += compute_fft_features(video_obj.rightShoulder_y)
    feature_vector += compute_fft_features(video_obj.leftElbow_x)
    feature_vector += compute_fft_features(video_obj.leftElbow_y)
    feature_vector += compute_fft_features(video_obj.rightElbow_x)
    feature_vector += compute_fft_features(video_obj.rightElbow_y)
    feature_vector += compute_fft_features(video_obj.leftWrist_x)
    feature_vector += compute_fft_features(video_obj.leftWrist_y)
    feature_vector += compute_fft_features(video_obj.rightWrist_x)
    feature_vector += compute_fft_features(video_obj.rightWrist_y)

    #Variance
    feature_vector += np.std(video_obj.leftShoulder_x)
    feature_vector += np.std(video_obj.leftShoulder_y)
    feature_vector += np.std(video_obj.rightShoulder_x)
    feature_vector += np.std(video_obj.rightShoulder_y)
    feature_vector += np.std(video_obj.leftElbow_x)
    feature_vector += np.std(video_obj.leftElbow_y)
    feature_vector += np.std(video_obj.rightElbow_x)
    feature_vector += np.std(video_obj.rightElbow_y)
    feature_vector += np.std(video_obj.leftWrist_x)
    feature_vector += np.std(video_obj.leftWrist_y)
    feature_vector += np.std(video_obj.rightWrist_x)
    feature_vector += np.std(video_obj.rightWrist_y)

    features.append(feature_vector)

number_of_decomposed_features = 10
pca = PCA(number_of_decomposed_features, random_state=42)
scaled_feature_matrix = scale(features)
reduced_feature_matrix = pca.fit_transform(scaled_feature_matrix)
reduced_feature_matrix = reduced_feature_matrix[:, :number_of_decomposed_features]

print(reduced_feature_matrix)
print(len(reduced_feature_matrix),len(reduced_feature_matrix[0]))
