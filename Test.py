from preprocessing import Preprocess
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from classification import Classification
import pickle
from features import Features


preprocess = Preprocess()
preprocess.scale_points()

pose_objects = preprocess.new_pose_objects

features = []

features_obj = Features(pose_objects=pose_objects)
features_obj.compute_features()
features = features_obj.get_features()

number_of_decomposed_features = 20
pca = PCA(number_of_decomposed_features, random_state=42)
scaled_feature_matrix = scale(features)
reduced_feature_matrix = pca.fit_transform(scaled_feature_matrix)
reduced_feature_matrix = reduced_feature_matrix[:, :number_of_decomposed_features]

# print(reduced_feature_matrix)
# print(len(reduced_feature_matrix),len(reduced_feature_matrix[0]))

X = reduced_feature_matrix
# X = features
Y = [obj.label for obj in pose_objects]

print(len(X), len(Y))
clf_rforest = Classification('RForest', X, Y)
clf_rforest.get_classifier_object()
clf_rforest.get_metrics()
pickle.dump(clf_rforest.get_classifier(), open('RForest_model.pkl', 'wb'))
print()