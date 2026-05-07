import math
from src.torcs_client import Client

PI = math.pi
GEAR_SPEEDS = [0, 50, 80, 120, 150, 200]
ENABLE_TRACTION_CONTROL = True
CORNER_READING = 2.0
SLOW_DOWN_DISTANCE = 60
STRAIGHT_DISTANCE = 120
BRAKING_INTENSITY = 0.2
STEERING_EFFECT = 1.75
MAX_STEPS = 100000

def get_min_sensor_data(S):
    track = S.get('track', [200.0] * 19)
    left_sensors = track[:9]
    right_sensors = track[10:]
    return min(min(left_sensors), min(right_sensors))

def is_corner(S, min_reading):
    track = S.get('track', [200.0] * 19)
    speedX = S.get('speedX', 0)
    if min_reading < CORNER_READING or track[9] < speedX * 0.65:
        return True
    return False

def is_straight(current_speed, forward_length, target_speed):
    if current_speed >= (target_speed - 5) and forward_length > STRAIGHT_DISTANCE:
        return True
    return False

def hold_acceleration(S, safe_speed):
    min_sensor_data = get_min_sensor_data(S)
    speedX = S.get('speedX', 0)
    if is_corner(S, min_sensor_data) and speedX > safe_speed:
        return True
    return False

def slow_down(S):
    track = S.get('track', [200.0] * 19)
    speedX = S.get('speedX', 0)
    max_forwards_sensors = max(track[7:12])
    if max_forwards_sensors < speedX * 0.60:
        return True
    return False

def calculate_corner_speed(S, gentle_speed, sharp_speed):
    track = S.get('track', [200.0] * 19)
    max_forwards_sensors = max(track[8:11])
    safe_speed = gentle_speed
    if max_forwards_sensors < SLOW_DOWN_DISTANCE:
        safe_speed = sharp_speed
    return safe_speed

def calculate_steering(S, steer_gain, centering_gain):
    angle = S.get('angle', 0)
    trackPos = S.get('trackPos', 0)
    track = S.get('track', [200.0] * 19)
    
    steer = (angle * steer_gain / PI) - (trackPos * centering_gain)

    if is_corner(S, get_min_sensor_data(S)):
        left_avg = sum(track[:9]) / 8
        right_avg = sum(track[10:]) / 8
        bias = right_avg - left_avg
        
        # Invece di un "salto" fisso a 0.46 che fa sbacchettare (destra-sinistra),
        # usiamo una proporzione dolce. Se right_avg > left_avg, c'è più spazio a destra,
        # quindi sterziamo verso destra (valore negativo).
        steer -= bias * 0.005

    return max(-1.0, min(1.0, steer))

def calculate_throttle(S, R_dict, target_speed, straight_speed, gentle_speed, sharp_speed):
    speedX = S.get('speedX', 0)
    track = S.get('track', [200.0] * 19)
    current_steer = R_dict.get('steer', 0.0)
    current_accel = R_dict.get('accel', 0.0)
    
    t_speed = target_speed
    safe_speed = calculate_corner_speed(S, gentle_speed, sharp_speed)

    # Se siamo in rettilineo e lontani dalla target speed, usiamo il gas massimo.
    if is_straight(speedX, track[9], target_speed):
        t_speed = straight_speed

    if speedX < t_speed - (abs(current_steer) * STEERING_EFFECT):
        accel = min(1.0, current_accel + 0.6)  # Accelerazione più aggressiva in trazione
    else:
        accel = max(0.0, current_accel - 0.1)

    if hold_acceleration(S, safe_speed):
        accel = max(0.0, current_accel - 0.2)

    if speedX < 10:
        accel += 1.0 / (speedX + 0.1)

    return max(0.0, min(1.0, accel))

def apply_brakes(S, brake_threshold):
    angle = S.get('angle', 0)
    brake = 0.0
    if abs(angle) > brake_threshold:
        brake = BRAKING_INTENSITY
    if slow_down(S):
        brake += 0.1
    return min(1.0, brake)
    
def shift_gears(S):
    speedX = S.get('speedX', 0)
    gear = 1
    for i, speed in enumerate(GEAR_SPEEDS):
        if speedX > speed:
            gear = i + 1
    return min(gear, 6)

def traction_control(S, accel):
    if ENABLE_TRACTION_CONTROL:
        wheel_spin = S.get('wheelSpinVel', [0,0,0,0])
        if len(wheel_spin) == 4:
            if ((wheel_spin[2] + wheel_spin[3]) - (wheel_spin[0] + wheel_spin[1])) > 2:
                accel -= 0.1
    return max(0.0, accel)

def drive_modular(S, R_dict, params):
    R_dict['steer'] = calculate_steering(S, params["steer_gain"], params["centering_gain"])
    R_dict['accel'] = calculate_throttle(S, R_dict, params["target_speed"], params["straight_speed"], 
                                         params["gentle_speed"], params["sharp_speed"])
    R_dict['brake'] = apply_brakes(S, params["brake_threshold"])
    R_dict['accel'] = traction_control(S, R_dict['accel'])
    R_dict['gear'] = shift_gears(S)

def run_lap(logger):
    params = logger.mix_values()
    if params is None:
        print("No params to run. Check recordings.csv")
        return

    print(f"Running experimental lap with params: {params}")

    C = Client(port=3001)
    
    stuck_timer = 0
    offtrack_count = 0
    max_dist_raced = 0
    completed = False
    
    # Run the race loop
    for step in range(MAX_STEPS, 0, -1):
        if not C.get_servers_input():
            break
            
        S = C.S.d
        R = C.R.d
        
        drive_modular(S, R, params)
        C.respond_to_server()
        
        # Tracking stats
        distRaced = S.get('distRaced', 0)
        speedX = S.get('speedX', 0)
        trackPos = S.get('trackPos', 0)
        
        if distRaced > max_dist_raced:
            max_dist_raced = distRaced
            
        if abs(trackPos) > 1.0:
            offtrack_count += 1
            
        # Check if stuck
        if distRaced > 10 and speedX < 5:
            stuck_timer += 1
        else:
            stuck_timer = 0
            
        if stuck_timer > 100: # Stuck for ~2 seconds
            print("Car got stuck! Aborting lap.")
            params['lap_time'] = 10000000.0
            break

        # Check lap completion (Assume ~3608m track or server shutdown)
        lastLapTime = S.get('lastLapTime', 0)
        if lastLapTime > 0 and distRaced > 3000:
            params['lap_time'] = lastLapTime
            completed = True
            print(f"Lap completed in {lastLapTime}s")
            break
            
        # Fallback if track length is known (Aalborg is ~2587m, CG-Track-1 is ~3100m, etc)
        # Using a generic large distance or wait for server to cut
        if distRaced >= 3608.45:
            params['lap_time'] = S.get('curLapTime', 10000000.0)
            completed = True
            break
            
    C.shutdown()
    
    params['completed'] = completed
    params['damage'] = C.S.d.get('damage', 0.0)
    params['offtrack_count'] = offtrack_count
    
    if not completed and params.get('lap_time', 0) == 0:
        params['lap_time'] = 10000000.0

    logger.log_data(params)

if __name__ == "__main__":
    from logger import LogData
    run_lap(LogData())
