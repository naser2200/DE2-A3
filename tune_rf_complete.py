import ray
from ray import tune
from ray.tune import run_experiments
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Step 3: Configure the search space
config = {
    "max_depth": tune.randint(1, 20),  # Sample values from 1 to 20
    "n_estimators": tune.randint(10, 200),  # Sample values from 10 to 200
    "ccp_alpha": tune.loguniform(1e-6, 1e-2),  # Sample values from a log-scaled distribution
}


# Step 4: Implement the train function
def train_rf(config):
    iris = load_iris()
    X, y = iris.data, iris.target
    model = RandomForestClassifier(
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        ccp_alpha=config["ccp_alpha"],
        random_state=42,
    )
    score = cross_val_score(model, X, y, cv=3).mean()
    tune.report(mean_accuracy=score)

# Step 5: Execute the pipeline

# Initialize Ray
ray.init(address="auto")

# Run experiments
run_experiments({
    "rf_hyperparameter_tuning": {
        "run": train_rf,
        "config": config,
        "num_samples": 50,
        "resources_per_trial": {"cpu": 2},
        "stop": {"mean_accuracy": 0.99},
    }
})

# Analyze the results and get the best trial
analysis = ExperimentAnalysis("ray_results/rf_hyperparameter_tuning")
best_trial = analysis.get_best_trial(metric="mean_accuracy", mode="max")
best_config = best_trial.config

# Print the best trial config and mean accuracy
print("Best trial config: ", best_config)
print("Best trial mean accuracy: ", best_trial.last_result["mean_accuracy"])

# Train the RandomForestClassifier model with the best hyperparameters
iris = load_iris()
X, y = iris.data, iris.target
best_model = RandomForestClassifier(
    max_depth=best_config["max_depth"],
    n_estimators=best_config["n_estimators"],
    ccp_alpha=best_config["ccp_alpha"],
    random_state=42,
)
best_model.fit(X, y)
