import math
from src.torcs_client import Client
from logger import LogData
from experimental import drive_modular

MAX_STEPS = 100000

def run_fastest():
    logger = LogData()
    best_params = logger.get_best_params()
    
    if best_params is None:
        print("No best params found. Check recordings.csv")
        return

    print("=== STARTING FASTEST DRIVER ===")
    print("Using optimal params:")
    for k, v in best_params.items():
        if k not in ['run_id', 'lap_time', 'completed', 'damage', 'offtrack_count']:
            print(f"  {k}: {v}")

    C = Client(port=3001)
    
    # Run the race loop
    for step in range(MAX_STEPS, 0, -1):
        if not C.get_servers_input():
            break
            
        S = C.S.d
        R = C.R.d
        
        drive_modular(S, R, best_params)
        C.respond_to_server()
            
    C.shutdown()
    print("Run finished.")

if __name__ == "__main__":
    run_fastest()
