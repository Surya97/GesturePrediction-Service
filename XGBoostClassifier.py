from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


class XGBoost:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.clf = XGBClassifier()
        self.grid_search_params = {'nthread': [4],
                                   'objective': ['binary:logistic'],
                                   'learning_rate': [0.05],
                                   'max_depth': [6],
                                   'min_child_weight': [11],
                                   'silent': [1],
                                   'subsample': [0.8],
                                   'colsample_bytree': [0.7],
                                   'n_estimators': [100],
                                   'missing': [-999],
                                   'seed': [1337]}

        self.clf = GridSearchCV(self.clf, self.grid_search_params, cv=10)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_Test = y_test
        self.y_pred = None

    def get_classifier(self):
        return self.clf

    def train(self):
        self.clf.fit(self.x_train, self.y_train)

    def predict(self, x_test=None):
        self.y_pred = self.clf.predict(self.x_test)
        return self.y_pred
