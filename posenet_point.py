class PoseNetPoint:
    def __init__(self, label):
        self.label = label
        self.leftShoulder_x = []
        self.rightShoulder_x = []
        self.leftElbow_x = []
        self.rightElbow_x = []
        self.leftWrist_x = []
        self.rightWrist_x = []
        self.leftShoulder_y = []
        self.rightShoulder_y = []
        self.leftElbow_y = []
        self.rightElbow_y = []
        self.leftWrist_y = []
        self.rightWrist_y = []
        self.leftHip_y = []
        self.nose_y = []
        self.knee_y = []

    def parse_json_data(self, data):
        for pose in data:
            keypoint = pose["keypoints"]
            for point in keypoint:
                if point["part"] == "nose":
                    self.nose_y.append(round(point["position"]["y"], 3))
                elif point["part"] == "leftKnee":
                    self.knee_y.append(round(point["position"]["y"], 3))
                elif point["part"] == "leftHip":
                    self.leftHip_y.append(round(point["position"]["y"], 3))
                elif point["part"] == "leftShoulder":
                    self.leftShoulder_x.append(round(point["position"]["x"], 3))
                    self.leftShoulder_y.append(round(point["position"]["y"], 3))
                elif point["part"] == "rightShoulder":
                    self.rightShoulder_x.append(round(point["position"]["x"], 3))
                    self.rightShoulder_y.append(round(point["position"]["y"], 3))
                elif point["part"] == "leftElbow":
                    self.leftElbow_x.append(round(point["position"]["x"], 3))
                    self.leftElbow_y.append(round(point["position"]["y"], 3))
                elif point["part"] == "rightElbow":
                    self.rightElbow_x.append(round(point["position"]["x"], 3))
                    self.rightElbow_y.append(round(point["position"]["y"], 3))
                elif point["part"] == "leftWrist":
                    self.leftWrist_x.append(round(point["position"]["x"], 3))
                    self.leftWrist_y.append(round(point["position"]["y"], 3))
                elif point["part"] == "rightWrist":
                    self.rightWrist_x.append(round(point["position"]["x"], 3))
                    self.rightWrist_y.append(round(point["position"]["y"], 3))
