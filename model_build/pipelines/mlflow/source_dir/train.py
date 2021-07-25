import os
import logging
import argparse
import numpy as np
import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # MLflow related parameters
    parser.add_argument("--tracking_uri", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--registered_model_name", type=str)
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument('--n-estimators', type=int, default=10)
    parser.add_argument('--min-samples-leaf', type=int, default=3)
    # input, feature list, and target
    parser.add_argument('--input', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--train-file', type=str, default='boston_train.csv')
    parser.add_argument('--test-file', type=str, default='boston_test.csv')
    parser.add_argument('--features', type=str)  # we ask user to explicitly name features
    parser.add_argument('--target', type=str)  # we ask user to explicitly name the target

    args, _ = parser.parse_known_args()

    logging.info('READING DATA')
    train_df = pd.read_csv(f'{args.input}/{args.train_file}')
    test_df = pd.read_csv(f'{args.input}/{args.train_file}')

    X_train = train_df[args.features.split()]
    X_test = test_df[args.features.split()]
    y_train = train_df[args.target]
    y_test = test_df[args.target]

    # set remote mlflow server
    logging.info('SET EXPERIMENT IN REMOTE MLFLOW SERVER')
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        params = {
            "n-estimators": args.n_estimators,
            "min-samples-leaf": args.min_samples_leaf,
            "features": args.features
        }
        mlflow.log_params(params)

        # TRAIN
        logging.info('TRAINING MODEL')
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # ABS ERROR AND LOG COUPLE PERF METRICS
        logging.info('EVALUATING MODEL')
        abs_err = np.abs(model.predict(X_test) - y_test)

        for q in [10, 50, 90]:
            logging.info(f'AE-at-{q}th-percentile: {np.percentile(a=abs_err, q=q)}')
            mlflow.log_metric(f'AE-at-{str(q)}th-percentile', np.percentile(a=abs_err, q=q))

        # SAVE MODEL
        # YOU CAN ADD A METRIC CONDITION HERE BEFORE REGISTERING THE MODEL
        logging.info('REGISTERING MODEL')
        # Make sure the IAM role has access to the MLflow bucket
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            registered_model_name=args.registered_model_name
        )
