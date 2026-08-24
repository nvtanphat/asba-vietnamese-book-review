import argparse
from ml.tuning.tuner import tune
p=argparse.ArgumentParser();p.add_argument("--model",required=True,choices=["logistic","linear_svm","textcnn","bilstm","phobert","xlmr","mdeberta","vit5"]);p.add_argument("--trials",type=int,default=20);a=p.parse_args();tune(a.model,a.trials)
