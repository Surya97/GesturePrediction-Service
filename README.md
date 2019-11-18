# CSE 535 - Mobile Computing
## Gesture Recognition Service Assignment

### Problem statement
Develop an online Application Service that accepts Human Pose Skeletal key points of a sign video and return the label of the sign as a JSON Response.
The Key points are generated using **Tensorflow's Po

### Feature Engineering
- Collected (x,y) co-ordinate of each part separately from the given PoseNet output files.
- Scaled the y-coordinates based on the mean distance between nose and hip.
- As the x-coordinates and y-coordinates of a particular part for the whole gesture is forming a signal, extracted signal based features.
- The following are the features considered
    - **Fast Fourier Transform** for x-coordinates and y-coordinates separately and merge them for a specific part and repeat the same for all parts
    - **Standard Deviation** for x-coordinates and y-coordinates separately for each part and merge them
    - **Mean** for x-coordinates and y-coordinates separately for each part and merge them.
    
- The final feature vector of a particular PoseNet file contained the above mentioned features.

### Model Training
- The dataset is split into train and test set in 70:30 ratio.
- The following models are considered as the final models
    - **Support Vector Machine (SVM)**
    - **Latent Discriminant Analysis**
    - **Logistic Regression**
    - **Random Forest**

We have achieved a maximum validation set accuracy of 69%.

### REST API
- We have used AWS Lambda service to host the service
- All the models are saved in pickled format and uploaded to S3 bucket.
- In the lambda function we tried to do preprocess the given JSON input and predict the label by using the models loaded from S3 bucket.
- Here is the link to the service [REST API](https://tlbms0lmhg.execute-api.us-east-1.amazonaws.com/default/gesture_prediction)


### Challenges faced
- Out of the given 6 gestures, there is no major change in the x-coordinates for *car* and *gift* gestures.
- There were different number of keypoints for different videos of the same gesture. So, huge amount of preprocessing was required.
- Also, most of the gestures given to classify have major movement in the fingers rather than shoulder/wrist/hand.
