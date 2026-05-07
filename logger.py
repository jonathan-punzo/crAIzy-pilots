import pandas as pd
import random
import os

class LogData:
    def __init__(self, path="recordings.csv"):
        self.path = path
        self.lt_label = "lap_time"
        
        # Define the parameter ranges for mutations
        self.param_ranges = {
            "target_speed": (75.0, 300.0),
            "steer_gain": (1.0, 100.0),
            "centering_gain": (0.0, 2.0),
            "brake_threshold": (0.0, 1.5),
            "gentle_speed": (85.0, 160.0),
            "sharp_speed": (1.0, 75.0),
            "straight_speed": (75.0, 300.0)
        }
        
        # Initialize if not exists
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                f.write("run_id,target_speed,steer_gain,centering_gain,brake_threshold,gentle_speed,sharp_speed,straight_speed,lap_time,completed,damage,offtrack_count\n")
                f.write("1,160.0,30.0,0.2,0.4,140.0,65.0,194.0,10000000.0,False,0.0,0\n")

    def get_data(self):
        try:
            data = pd.read_csv(self.path)
            if len(data) == 0:
                print("The file is empty!")
                return None
            return data
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None
    
    def get_best_params(self):
        """Returns the dictionary of the best parameters found so far based on lap_time."""
        data = self.get_data()
        if data is not None and not data.empty:
            fastest_lap = data[self.lt_label].min()
            # Get the most recent row that achieved this time
            best_row = data[data[self.lt_label] == fastest_lap].iloc[-1]
            return best_row.to_dict()
        return None

    def deviate_value(self, param_name):
        """Returns a random value within the specified range for the given parameter."""
        min_val, max_val = self.param_ranges[param_name]
        val = random.uniform(min_val, max_val)
        return round(val, 3)

    def clean_data(self, params):
        """Enforces logical constraints between parameters."""
        # Sharp corner speed should not be greater than gentle corner speed
        if params["sharp_speed"] > params["gentle_speed"]:
            params["gentle_speed"] = params["sharp_speed"]

        # Gentle corner speed should not be greater than target overall speed
        if params["gentle_speed"] > params["target_speed"]:
            params["target_speed"] = params["gentle_speed"]

        # Target speed should not be greater than straight line speed
        if params["target_speed"] > params["straight_speed"]:
            params["straight_speed"] = params["target_speed"]

        return params

    def mix_values(self):
        """Gets the best params, randomly mutates one, and returns the new parameter set."""
        best_params = self.get_best_params()
        if best_params is None:
            print("No baseline data available.")
            return None
            
        # Select a random parameter to mutate (excluding non-tunable columns)
        tunable_params = list(self.param_ranges.keys())
        param_to_mutate = random.choice(tunable_params)
        
        # Mutate the parameter
        best_params[param_to_mutate] = self.deviate_value(param_to_mutate)
        
        # Apply constraints
        new_params = self.clean_data(best_params)
        
        return new_params

    def log_data(self, lap_run_data_dict):
        """Appends a new run to the CSV."""
        data = self.get_data()
        
        # Calculate new run_id
        if data is not None and not data.empty:
            next_run_id = int(data["run_id"].max()) + 1
        else:
            next_run_id = 1
            
        lap_run_data_dict["run_id"] = next_run_id
        
        # Order columns to match standard
        columns = ["run_id", "target_speed", "steer_gain", "centering_gain", "brake_threshold", 
                   "gentle_speed", "sharp_speed", "straight_speed", "lap_time", 
                   "completed", "damage", "offtrack_count"]
                   
        # Build CSV line
        line = []
        for col in columns:
            val = lap_run_data_dict.get(col, 0)
            line.append(str(val))
            
        with open(self.path, "a") as f:
            f.write(",".join(line) + "\n")
            
        print(f"Logged Run {next_run_id}: LapTime {lap_run_data_dict.get('lap_time')} | Completed: {lap_run_data_dict.get('completed')}")
