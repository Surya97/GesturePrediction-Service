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
# reduced_feature_matrix = features_obj.compute_pca()

# print(reduced_feature_matrix)
# print(len(reduced_feature_matrix),len(reduced_feature_matrix[0]))

# X = reduced_feature_matrix
X = features_obj.get_features()
Y = [obj.label for obj in pose_objects]

print(len(X), len(Y))
clf_rforest = Classification('RForest', X, Y)
clf_rforest.get_classifier_object()
clf_rforest.get_metrics()
pickle.dump(clf_rforest.get_classifier(), open('RForest_model.pkl', 'wb'))
print()

clf_svm = Classification('svm', X, Y)
clf_svm.get_classifier_object()
clf_svm.get_metrics()
pickle.dump(clf_svm.get_classifier(), open('SVM_model.pkl', 'wb'))
print()

clf_lda = Classification('lda', X, Y)
clf_lda.get_classifier_object()
clf_lda.get_metrics()
pickle.dump(clf_lda.get_classifier(), open('LDA_model.pkl', 'wb'))

clf_dt = Classification('logreg', X, Y)
clf_dt.get_classifier_object()
clf_dt.get_metrics()
pickle.dump(clf_dt.get_classifier(), open('LogReg_model.pkl', 'wb'))
print()


