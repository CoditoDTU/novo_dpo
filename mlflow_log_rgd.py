import mlflow
import pandas as pd


files=["train_df_result_test_puma_2"]
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Rodri_model")

for file in files:
    df = pd.read_csv(f"{file}.csv")

    for _, row in df.iterrows():
        run_name = str(row["id"])
        print(run_name)

        with mlflow.start_run(run_name=run_name):
            # Log parameters
            param_keys = [
                'n_epochs', 'betas', 'epsilon', 'learning_rate', 'P_train',
                'N_pairs', 'N_pairs_val_1', 'N_pairs_val_'
            ]
            for key in param_keys:
                if key in df.columns and pd.notnull(row[key]):
                    mlflow.log_param(key, row[key])

            # Log metrics
            metric_keys = ['Eval_score_1', 'Eval_score_2']
            for key in metric_keys:
                if key in df.columns and pd.notnull(row[key]):
                    mlflow.log_metric(key, row[key])
