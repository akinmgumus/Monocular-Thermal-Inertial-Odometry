import os
import subprocess
import sys

def run_test(script_name, method, use_clahe='0', dl_ver='1'):
    env = os.environ.copy()
    env['TVIO_METHOD'] = method
    env['TVIO_USE_CLAHE'] = use_clahe
    env['TVIO_DL_VER'] = dl_ver
    env['TVIO_LIVE_PLOT'] = '0'  # Disable live plot for automation
    
    cmd = [sys.executable, script_name]
    
    print("-" * 60)
    print(f"Running: {script_name}")
    print(f"  Method: {method.upper()}")
    print(f"  CLAHE: {'ON' if use_clahe == '1' else 'OFF'}")
    print(f"  DL_VER: {dl_ver}")
    print("-" * 60)
    
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"WARNING: {script_name} failed with exit code {result.returncode}")
    else:
        print(f"SUCCESS: {script_name} finished successfully.\n")

def main():
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define permutations
    methods = ['klt', 'orb']
    clahes = ['1', '0']
    dl_vers = ['1', '2']
    
    run_counter = 1
    total_runs = 30
    
    print(f"Starting automated test suite. Total runs planned: {total_runs}\n")

    # 1. EuRoC (2 runs)
    script_euroc = os.path.join(scripts_dir, 'test_msckf_vio_euroc.py')
    for m in methods:
        print(f"=== Run {run_counter}/{total_runs} ===")
        run_test(script_euroc, method=m)
        run_counter += 1

    # 2. FIReStereo, ROVTIO, SThereo (24 runs)
    thermal_scripts = ['test_msckf_vio_firestereo.py', 'test_msckf_vio_rovtio.py', 'test_msckf_vio_sthereo.py']
    for script_base in thermal_scripts:
        script_path = os.path.join(scripts_dir, script_base)
        for m in methods:
            for c in clahes:
                for dl in dl_vers:
                    print(f"=== Run {run_counter}/{total_runs} ===")
                    run_test(script_path, method=m, use_clahe=c, dl_ver=dl)
                    run_counter += 1

    # 3. Voxlbag / AerialTN (4 runs)
    script_voxlbag = os.path.join(scripts_dir, 'test_msckf_vio_voxlbag.py')
    for m in methods:
        for c in clahes:
            print(f"=== Run {run_counter}/{total_runs} ===")
            run_test(script_voxlbag, method=m, use_clahe=c, dl_ver='1')
            run_counter += 1

    print("\n" + "=" * 60)
    print("ALL 30 RUNS COMPLETED.")
    print("Check results/summary_table.png and results/all_runs.csv for final outcomes.")
    print("=" * 60)

if __name__ == "__main__":
    main()
