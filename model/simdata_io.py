# model/simdata_io.py
import pickle
import os

def save_simulation_data(trajectory_data, tars, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "simdata.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump({
            "trajectory_data": trajectory_data,
            "targets": tars
        }, f)
    print(f"Simulation data saved to {file_path}")

def load_simulation_data(data_dir):
    file_path = os.path.join(data_dir, 'simdata.pkl')
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data["trajectory_data"], data["targets"]

