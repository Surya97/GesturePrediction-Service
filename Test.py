from train_preprocessing import Preprocess
from numpy.fft import fft
import numpy as np


def compute_fft_features(self, label_vector):
    feature_vector = []
    max_points_per_series = 4
    cgm_fft = np.abs(fft(label_vector))
    cgm_fft = cgm_fft.tolist()
    cgm_fft.sort(reverse=True)
    feature_vector += cgm_fft[:max_points_per_series]
    return feature_vector

preprocess = Preprocess()
preprocess.scale_points()
print(preprocess.new_pose_objects[0].leftShoulder_y)

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


    features.append(feature_vector)