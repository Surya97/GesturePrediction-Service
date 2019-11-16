from preprocessing import Preprocess
from classification import Classification
import pickle
from features import Features
import json
import os

preprocess = Preprocess()
preprocess.scale_points()

pose_objects = preprocess.new_pose_objects

features = []

features_obj = Features(pose_objects=pose_objects)
features_obj.compute_features()
reduced_feature_matrix = features_obj.compute_pca()

# print(reduced_feature_matrix)
# print(len(reduced_feature_matrix),len(reduced_feature_matrix[0]))

X = reduced_feature_matrix
Y = [obj.label for obj in pose_objects]

print(len(X), len(Y))
clf_rforest = Classification('RForest', X, Y)
clf_rforest.get_classifier_object()
clf_rforest.get_metrics()
pickle.dump(clf_rforest.get_classifier(), open('RForest_model.pkl', 'wb'))
print()
