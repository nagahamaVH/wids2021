import xgboost as xgb
import pickle
import os
from tools import make_submission

DATA_PATH = "./data/data_remove_redudant.pkl"
TEST_PATH = "./data/UnlabeledWiDS2021.csv"
SUBMISSION_PATH = "./submissions"

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]

train_dmatrix = xgb.DMatrix(x_train, label=y_train)
test_dmatrix = xgb.DMatrix(x_test)

params = {"objective": "binary:logistic",
          "max_depth": 20,
          "max_leaves": 15,
          "tree_method": "gpu_hist"}

model = xgb.train(params, train_dmatrix, num_boost_round=120)

pred = model.predict(test_dmatrix)

sub_name = os.path.join(SUBMISSION_PATH, "xgb_remove_redundant_auc.csv")
make_submission(pred, TEST_PATH, sub_name)
