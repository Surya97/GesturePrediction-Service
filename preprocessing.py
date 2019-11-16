import os
from posenet_point import PoseNetPoint
import json
import numpy as np


class Preprocess:
    def __init__(self, json_data=None):
        self.labels = ["book", "car", "gift", "movie", "sell", "total"]
        self.poseObjects = []
        self.mean_nose = None
        self.mean_hip = None
        self.read_data(json_data)
        self.new_pose_objects = []
        self.max_left = -10000
        self.min_right= 10000

    def read_data(self, json_data=None):
        if json_data is None:
            for label in self.labels:
                label_files = os.listdir('./data/' + label)
                label_file_paths = []
                for file in label_files:
                    label_file_paths.append(os.path.join('./data/', label, file))

                for file in label_file_paths:
                    with open(file, encoding="utf-8") as label_json_file:
                        json_data = json.load(label_json_file)
                        posenet_point = PoseNetPoint(label)
                        posenet_point.parse_json_data(json_data)
                        self.poseObjects.append(posenet_point)
        else:
            json_data = json.load(json_data)
            posenet_point = PoseNetPoint(label=None)
            posenet_point.parse_json_data(json_data)
            self.poseObjects.append(posenet_point)

    def calculate_mean(self):
        sum_nose = 0
        sum_hip = 0
        count_nose = 0
        count_hip = 0

        for pose_object in self.poseObjects:
            count_hip += len(pose_object.leftHip_y)
            sum_hip += sum(pose_object.leftHip_y)

            count_nose += len(pose_object.nose_y)
            sum_nose += sum(pose_object.nose_y)

        self.mean_hip = sum_hip / count_hip
        self.mean_nose = sum_nose / count_nose

    def calc_lr_boundaries(self):
        for pose_object in self.poseObjects:
            self.max_left = max(self.max_left, max(max(pose_object.leftWrist_x), max(pose_object.leftElbow_x)))
            self.min_right = min(self.min_right, min(min(pose_object.rightWrist_x), min(pose_object.rightElbow_x)))
        return

    def scale_points(self, calculate_scale=True):
        if calculate_scale:
            self.calculate_mean()
            self.calc_lr_boundaries()
        else:
            self.mean_nose = 516.488
            self.mean_hip = 1310.978
        for pose_object in self.poseObjects:
            nose_y = pose_object.nose_y
            hip_y = pose_object.leftHip_y

            left_shoulder_y = pose_object.leftShoulder_y
            right_shoulder_y = pose_object.rightShoulder_y
            left_elbow_y = pose_object.leftElbow_y
            right_elbow_y = pose_object.rightElbow_y
            left_wrist_y = pose_object.leftWrist_y
            right_wrist_y = pose_object.rightWrist_y

            left_shoulder_x = pose_object.leftShoulder_x
            right_shoulder_x = pose_object.rightShoulder_x
            left_elbow_x = pose_object.leftElbow_x
            right_elbow_x = pose_object.rightElbow_x
            left_wrist_x = pose_object.leftWrist_x
            right_wrist_x = pose_object.rightWrist_x

            new_left_shoulder_y = []
            new_right_shoulder_y = []
            new_left_elbow_y = []
            new_right_elbow_y = []
            new_left_wrist_y = []
            new_right_wrist_y = []

            new_left_shoulder_x = []
            new_right_shoulder_x = []
            new_left_elbow_x = []
            new_right_elbow_x = []
            new_left_wrist_x = []
            new_right_wrist_x = []

            for i in range(len(left_shoulder_y)):
                new_left_shoulder_y.append(self.mean_nose + (self.mean_hip - self.mean_nose)
                                           * ((left_shoulder_y[i] - nose_y[i])/(hip_y[i] - nose_y[i])))

                new_right_shoulder_y.append(self.mean_nose + (self.mean_hip - self.mean_nose)
                                            * ((right_shoulder_y[i] - nose_y[i]) / (hip_y[i] - nose_y[i])))

                new_left_elbow_y.append(self.mean_nose + (self.mean_hip - self.mean_nose)
                                        * ((left_elbow_y[i] - nose_y[i]) / (hip_y[i] - nose_y[i])))

                new_right_elbow_y.append(self.mean_nose + (self.mean_hip - self.mean_nose)
                                         * ((right_elbow_y[i] - nose_y[i]) / (hip_y[i] - nose_y[i])))

                new_left_wrist_y.append(self.mean_nose + (self.mean_hip - self.mean_nose)
                                        * ((left_wrist_y[i] - nose_y[i]) / (hip_y[i] - nose_y[i])))

                new_right_wrist_y.append(self.mean_nose + (self.mean_hip - self.mean_nose)
                                         * ((right_wrist_y[i] - nose_y[i]) / (hip_y[i] - nose_y[i])))

                #x- axis
                new_left_shoulder_x.append(self.min_right + (
                            (left_shoulder_x[i] - min(min(right_wrist_x[i], right_shoulder_x[i]), right_elbow_x[i])) / (
                                max(max(left_wrist_x[i], left_shoulder_x[i]), left_elbow_x[i]) - min(right_wrist_x[i],
                                                                                                     right_elbow_x[
                                                                                                         i]))))
                new_right_shoulder_x.append(self.min_right + (
                        (right_shoulder_x[i] - min(min(right_wrist_x[i], right_shoulder_x[i]), right_elbow_x[i])) / (
                        max(max(left_wrist_x[i], left_shoulder_x[i]), left_elbow_x[i]) - min(right_wrist_x[i],
                                                                                             right_elbow_x[
                                                                                                 i]))))
                new_left_elbow_x.append(self.min_right + (
                        (left_elbow_x[i] - min(min(right_wrist_x[i], right_shoulder_x[i]), right_elbow_x[i])) / (
                        max(max(left_wrist_x[i], left_shoulder_x[i]), left_elbow_x[i]) - min(right_wrist_x[i],
                                                                                             right_elbow_x[
                                                                                                 i]))))
                new_right_elbow_x.append(self.min_right + (
                        (right_elbow_x[i] - min(min(right_wrist_x[i], right_shoulder_x[i]), right_elbow_x[i])) / (
                        max(max(left_wrist_x[i], left_shoulder_x[i]), left_elbow_x[i]) - min(right_wrist_x[i],
                                                                                             right_elbow_x[
                                                                                                 i]))))
                new_left_wrist_x.append(self.min_right + (
                        (left_wrist_x[i] - min(min(right_wrist_x[i], right_shoulder_x[i]), right_elbow_x[i])) / (
                        max(max(left_wrist_x[i], left_shoulder_x[i]), left_elbow_x[i]) - min(right_wrist_x[i],
                                                                                             right_elbow_x[
                                                                                                 i]))))

                new_right_wrist_x.append(self.min_right + (
                        (right_wrist_x[i] - min(min(right_wrist_x[i], right_shoulder_x[i]), right_elbow_x[i])) / (
                        max(max(left_wrist_x[i], left_shoulder_x[i]), left_elbow_x[i]) - min(right_wrist_x[i],
                                                                                             right_elbow_x[
                                                                                                 i]))))
            #x - axis

            #print(self.min_right + (self.max_left - self.min_right)*((left_shoulder_x[0]-min(right_wrist_x[0],right_elbow_x[0]))/(max(left_wrist_x[0],left_elbow_x[0])-min(right_wrist_x[0],right_elbow_x[0]))))


            new_pose_object = PoseNetPoint(pose_object.label)

            new_pose_object.leftShoulder_x = new_left_shoulder_x
            new_pose_object.rightShoulder_x = new_right_shoulder_x
            new_pose_object.leftElbow_x = new_left_elbow_x
            new_pose_object.rightElbow_x = new_right_elbow_x
            new_pose_object.leftWrist_x = new_left_wrist_x
            new_pose_object.rightWrist_x = new_right_wrist_x

            new_pose_object.leftShoulder_y = new_left_shoulder_y
            new_pose_object.rightShoulder_y = new_right_shoulder_y
            new_pose_object.leftElbow_y = new_left_elbow_y
            new_pose_object.rightElbow_y = new_right_elbow_y
            new_pose_object.leftWrist_y = new_left_wrist_y
            new_pose_object.rightWrist_y = new_right_wrist_y

            self.new_pose_objects.append(new_pose_object)



