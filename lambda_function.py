import json
import datetime
from preprocessing import Preprocess
from classification import Classification
import pickle
from features import Features
import boto3


def lambda_handler(event, context):
    # TODO implement
    
    json_data = json.loads(event['body'])
    preprocess = Preprocess(json_data=json_data)
    preprocess.scale_points(calculate_scale=False)

    pose_objects = preprocess.new_pose_objects

    features = []

    features_obj = Features(pose_objects=pose_objects)
    features_obj.compute_features()
    features = features_obj.get_features()
    # pca_model = pickle.load(open('pca.pkl', 'rb'))
    # reduced_feature_matrix = pca_model.transform(features)

    s3 = boto3.resource('s3')

    random_forest_classifier = pickle.loads(s3.Bucket("gesture-recognition").Object("RForest_model.pkl").get()['Body'].read())

    svm_classifier = pickle.loads(s3.Bucket("gesture-recognition").Object("SVM_model.pkl").get()['Body'].read())

    knn_classifier = pickle.loads(s3.Bucket("gesture-recognition").Object("KNN_model.pkl").get()['Body'].read())

    logreg_classifier = pickle.loads(s3.Bucket("gesture-recognition").Object("LogReg_model.pkl").get()['Body'].read())

    prediction_rf = random_forest_classifier.predict(features)
    prediction_svm = svm_classifier.predict(features)
    prediction_knn = knn_classifier.predict(features)
    prediction_logreg = logreg_classifier.predict(features)

    data = {
        "1": prediction_rf[0],
        "2": prediction_svm[0],
        "3": prediction_knn[0],
        "4": prediction_logreg[0]
    }
    return {
        'statusCode': 200,
        'body': json.dumps(data)
    }
