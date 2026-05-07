import subprocess
import pyautogui
import time
import os
import sys

# IMPOSTAZIONI WINDOWS
TORCS_EXE = r"C:\Users\jonat\Desktop\torcs\torcs\wtorcs.exe"
WORKING_DIRECTORY_PATH = r"C:\Users\jonat\Desktop\torcs\torcs"
CLICK_X = 400 # Da calibrare se il mouse non clicca su "Race" (Centro-Alto dello schermo di TORCS)
CLICK_Y = 150 # Da calibrare se il mouse non clicca su "Race"
STEPS = 5 # Numero di iterazioni del tuning loop

def start_lap():
    print("Avvio di TORCS...")
    
    # 1. Killa eventuale wtorcs.exe precedente
    os.system("taskkill /F /IM wtorcs.exe 2>NUL")
    time.sleep(1)

    # 2. Avvia wtorcs
    if not os.path.exists(TORCS_EXE):
        print(f"ERRORE: Impossibile trovare {TORCS_EXE}. Modifica il file startup.py con il percorso corretto.")
        sys.exit(1)
        
    proc = subprocess.Popen([TORCS_EXE, "-nofuel", "-nodamage", "-nolaptime"], cwd=WORKING_DIRECTORY_PATH)
    time.sleep(3) # Attendi caricamento menu
    
    # 3. Naviga nei menu (Esempio: due click per Quick Race -> Start)
    pyautogui.moveTo(CLICK_X, CLICK_Y)
    pyautogui.doubleClick()
    time.sleep(0.5)
    pyautogui.click()

    # Opzionale: Accelerare simulazione con + o configurare visuale
    with pyautogui.hold('shift'):
        pyautogui.press('+', presses=3) # Velocizza il gioco se permesso
        
    print("TORCS avviato e configurato, lancio experimental.py...")
    
    # 4. Lancia experimental.py come processo
    try:
        subprocess.run([sys.executable, "experimental.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione di experimental.py: {e}")
    finally:
        # Pulisci al termine
        print("Terminata iterazione, chiudo TORCS...")
        proc.kill()
        time.sleep(2)

if __name__ == "__main__":
    print(f"--- INIZIO TUNING LOOP: {STEPS} ITERAZIONI ---")
    try:
        for x in range(STEPS):
            print(f"\n--- Iterazione {x+1}/{STEPS} ---")
            start_lap()
    except KeyboardInterrupt:
        print("\nTuning loop interrotto dall'utente.")
        os.system("taskkill /F /IM wtorcs.exe 2>NUL")
