from preprocessing import Preprocess
from classification import Classification
import pickle
from features import Features
import json
import os
# preprocess = Preprocess()
# preprocess.scale_points()
#
# pose_objects = preprocess.new_pose_objects
#
# features = []
#
# features_obj = Features(pose_objects=pose_objects)
# features_obj.compute_features()
# reduced_feature_matrix = features_obj.compute_pca()
#
# # print(reduced_feature_matrix)
# # print(len(reduced_feature_matrix),len(reduced_feature_matrix[0]))
#
# X = reduced_feature_matrix
# Y = [obj.label for obj in pose_objects]
#
# print(len(X), len(Y))
# clf_rforest = Classification('RForest', X, Y)
# clf_rforest.get_classifier_object()
# clf_rforest.get_metrics()
# pickle.dump(clf_rforest.get_classifier(), open('RForest_model.pkl', 'wb'))
# print()
files = os.listdir('./data/gift')
for file in files:
    file_path = os.path.join('./data/gift', file)
    with open(file_path, encoding="utf-8") as data:
        json_data = json.load(data)

    # print(json_data)

    preprocess = Preprocess(json_data=json_data)
    preprocess.scale_points(calculate_scale=False)

    pose_objects = preprocess.new_pose_objects

    features = []

    features_obj = Features(pose_objects=pose_objects)
    features_obj.compute_features()
    features = features_obj.get_features()
    pca_model = pickle.load(open('pca.pkl', 'rb'))
    # reduced_feature_matrix = pca_model.transform(features)

    random_forest_classifier = pickle.load(open('RForest_model.pkl', 'rb'))

    # prediction = random_forest_classifier.predict(reduced_feature_matrix)
    prediction = random_forest_classifier.predict(features)
    print('Prediction', prediction)
