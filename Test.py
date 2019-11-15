from train_preprocessing import Preprocess


preprocess = Preprocess()
preprocess.scale_points()
print(preprocess.new_pose_objects[0].leftShoulder_y)
