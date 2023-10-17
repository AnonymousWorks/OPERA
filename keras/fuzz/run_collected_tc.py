import os
import sys
import subprocess
import multiprocessing
import datetime

import warnings
warnings.filterwarnings('ignore')

keras_run_head = """from test_tvm_keras import layer_test
from tensorflow import keras
import tensorflow as tf
import numpy as np

"""


def run_subprocess(python_program):
    print(f"Execute subprocess: {python_program}")
    run_flag = subprocess.run(['python', python_program], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if run_flag.returncode == 0:  # run well
        output = run_flag.stdout.decode('utf-8')
        output_final = ''
        for line in output.split("\\n"):
            output_final += line
        print(output_final)
        return
    else:
        err_output = run_flag.stderr.decode('utf-8')
        output_final = ''
        for line in err_output.split("\\n"):
            output_final += line

        print(f">>>> [Warning] Check the test case in file {python_program}")
        print(output_final)
    return


def gen_test_case_file(test_str, save_dir, test_id):
    test_str = keras_run_head + test_str
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.system(f"cp test_tvm_keras.py {save_dir}")
    save_test_file_path = os.path.join(save_dir, f'test_{test_id}.py')
    with open(save_test_file_path, 'w', encoding='utf-8') as test_f:
        test_f.write(test_str)
    return save_test_file_path


def run_all_test(test_file, begin_id=1):
    if 'docter' in test_file:
        begin_id = 20993
    with open(test_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    all_test_cases = []
    for cnt, line in enumerate(all_lines):
        line = line.strip()
        if line.startswith("layer_test"):
            line = line.strip()[:-1] + f"count={cnt+begin_id},)"  # give correct bug_id

            save_test_file_path = gen_test_case_file(line, "all_all", cnt+begin_id)
            all_test_cases.append(save_test_file_path)
            run_subprocess(save_test_file_path)

    #pool = multiprocessing.Pool(processes=1)#multiprocessing.cpu_count()//2)
    #pool.map(run_subprocess, all_test_cases)

    #pool.close()
    #pool.join()


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    # collected_test_cases_file = f"borrow_all_test.py"
    # collected_test_cases_file = f"borrow_all_test_deduplicate.py"
    collected_test_cases_file = sys.argv[1]
    run_all_test(collected_test_cases_file)

    endtime = datetime.datetime.now()
    print("Finish all, time consuming(min): ", (endtime - starttime).seconds/60)
