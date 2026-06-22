import os
import torch
import pandas as pd
import numpy as np
import warnings
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement

# ==========================================
# 0. Configuration
# ==========================================
FOLDER_NAME = "../data/optimization_results"
JSON_FILENAME = "qnehvi_ax_state.json"
CSV_FILENAME = "qnehvi_ax.csv"
RANDOM_SEED = 2026
# JSON_FILENAME = f"qnehvi_ax_state_seed{RANDOM_SEED}.json"
# CSV_FILENAME = f"qnehvi_ax_seed{RANDOM_SEED}.csv"

# Paths
os.makedirs(FOLDER_NAME, exist_ok=True)
JSON_PATH = os.path.join(FOLDER_NAME, JSON_FILENAME)
CSV_PATH = os.path.join(FOLDER_NAME, CSV_FILENAME)

# Random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Suppress FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 1. Search space
# ==========================================
search_space_parameters = [
    {"name": "C_aq", "type": "choice", "values": [0.1, 0.2, 0.3, 0.4, 0.5], "value_type": "float", "sort_values": True},
    {"name": "C_org", "type": "choice", "values": [0.01, 0.02, 0.03, 0.04, 0.05], "value_type": "float",
     "sort_values": True},
    {"name": "Phi", "type": "choice", "values": [0.3, 0.4, 0.5, 0.6, 0.7], "value_type": "float", "sort_values": True},
    {"name": "Q_total", "type": "choice", "values": [0.2, 0.4, 0.6, 0.8, 1.0], "value_type": "float",
     "sort_values": True},
    {"name": "L", "type": "choice", "values": [1.0, 2.0, 4.0], "value_type": "float", "sort_values": True},
]

# ==========================================
# 2. Generation strategy (Sobol + qNEHVI)
# ==========================================
gs = GenerationStrategy(
    steps=[
        GenerationStep(model=Models.SOBOL, num_trials=10, min_trials_observed=10),
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,
            # Ax automatically selects qNEHVI for the two-objective problem.
            model_kwargs={},
        ),
    ]
)


# ==========================================
# 3. Workflow
# ==========================================
def initialize_client():
    """Initialize AxClient and clean incompatible archived settings."""
    if os.path.exists(JSON_PATH):
        print(f"[LOAD] Loading existing state from {JSON_PATH} ...")
        client = AxClient.load_from_json_file(JSON_PATH)

        # =====================================================
        # Remove archived parameters that are incompatible with the active Ax version.
        # =====================================================
        try:
            steps = client.generation_strategy._steps
            for step in steps:
                if step.model_kwargs:
                    if "surrogate_spec" in step.model_kwargs:
                        print("[AUTO-FIX] Removing deprecated 'surrogate_spec' parameter...")
                        del step.model_kwargs["surrogate_spec"]

                    if "acquisition_class" in step.model_kwargs:
                        print("[AUTO-FIX] Removing explicit 'acquisition_class' parameter...")
                        del step.model_kwargs["acquisition_class"]
                        print("[AUTO-FIX] Default acquisition configuration restored.")

        except Exception as e:
            print(f"[WARNING] Archived configuration validation failed: {e}")

        return client
    else:
        print("[INIT] No existing state found; creating a new experiment...")
        client = AxClient(generation_strategy=gs)
        client.create_experiment(
            name="Microreactor_Hexanol_Opt",
            parameters=search_space_parameters,
            objectives={
                "yield": ObjectiveProperties(minimize=False, threshold=0.1),
                "sty": ObjectiveProperties(minimize=False, threshold=0.01),
            },
        )
        return client


def run_workflow():
    # 1. Load or initialize the client.
    client = initialize_client()

    # 2. Initialize the CSV if needed.
    if not os.path.exists(CSV_PATH):
        print("[Mode: Initialization] CSV not found; generating initial experiments...")

        init_data = []
        for i in range(10):
            params, trial_index = client.get_next_trial()
            row = params.copy()
            row['trial_index'] = trial_index
            row['yield'] = 0.0
            row['sty'] = 0.0
            init_data.append(row)

        df = pd.DataFrame(init_data)
        cols = ['trial_index'] + [col for col in df.columns if col != 'trial_index']
        df = df[cols]

        df.to_csv(CSV_PATH, index=False)
        client.save_to_json_file(JSON_PATH)

        print(f"[Success] Created {CSV_PATH}")
        print("Enter the results of the first 10 experiments in the CSV file.")
        return

    # 3. Read completed results.
    print("[Mode: Iteration] Reading CSV data...")
    df = pd.read_csv(CSV_PATH)

    pending_trials = df[(df['yield'] == 0) & (df['sty'] == 0)]

    if not pending_trials.empty:
        print(f"[Warning] {len(pending_trials)} CSV rows still have zero objective values.")
        print(f"Pending trial indices: {pending_trials['trial_index'].tolist()}")
        print("Complete the pending experiments before requesting another candidate.")
        return

    # 4. Synchronize CSV results with AxClient.
    new_data_added = False

    for index, row in df.iterrows():
        trial_idx = int(row['trial_index'])
        try:
            trial = client.experiment.trials[trial_idx]
            if trial.status.name == "RUNNING":
                print(f"[Update] Recording results for trial {trial_idx}...")
                client.complete_trial(
                    trial_index=trial_idx,
                    raw_data={
                        "yield": (row['yield'], 0.0),
                        "sty": (row['sty'], 0.0)
                    }
                )
                new_data_added = True
        except KeyError:
            pass

    if new_data_added:
        print("[Success] Results synchronized and model state updated.")
        client.save_to_json_file(JSON_PATH)
    else:
        print("[Info] All existing results are already synchronized.")

    # 5. Generate the next recommendation.
    print("Computing the next recommendation...")
    try:
        params, trial_index = client.get_next_trial()
    except Exception as e:
        print(f"[Error] Recommendation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"[New Trial] Generated trial {trial_index}")
    print(params)

    # 6. Append the new candidate to the CSV.
    new_row = params.copy()
    new_row['trial_index'] = trial_index
    new_row['yield'] = 0.0
    new_row['sty'] = 0.0

    new_df_row = pd.DataFrame([new_row])
    new_df_row = new_df_row[df.columns]

    new_df_row.to_csv(CSV_PATH, mode='a', header=False, index=False)
    client.save_to_json_file(JSON_PATH)

    print(f"[Success] Trial {trial_index} appended to the CSV file.")
    print("Run the experiment, enter its results, and execute this script again.")


if __name__ == "__main__":
    run_workflow()
