import os
import sys
import subprocess
import multiprocessing
import datetime
import shutil

import warnings
warnings.filterwarnings('ignore')


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
    else:  # invalid test cases
        err_output = run_flag.stderr.decode('utf-8')
        output_final = ''
        for line in err_output.split("\\n"):
            output_final += line

        print(f">>>> [Warning] Check the test case in file {python_program}")
        print(output_final)
    return


def gen_test_case_file(SUT, test_str, save_dir, test_id):
    run_head = f"""from test_{SUT}_keras import layer_test
from tensorflow import keras
import tensorflow as tf
import numpy as np

"""
    test_str = run_head + test_str
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy(f"test_{SUT}_keras.py", save_dir)

    save_test_file_path = os.path.join(save_dir, f'test_{test_id}.py')
    if not os.path.exists(save_test_file_path):
        with open(save_test_file_path, 'w', encoding='utf-8') as test_f:
            test_f.write(test_str)
    return save_test_file_path


def run_all_test(test_file, SUT, begin_id=1):
    with open(test_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    all_test_files = []
    for cnt, line in enumerate(all_lines):
        tc_id = cnt+begin_id
        line = line.strip()
        if line.startswith("layer_test"):
            tc_line = line.strip()[:-1] + f"count={tc_id},)"  # give correct bug_id
            save_test_file_path = gen_test_case_file(SUT, tc_line, "all_keras_tc", tc_id)
            all_test_files.append(save_test_file_path)
            # execute the all test cases
            # run_subprocess(save_test_file_path)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()//2)
    pool.map(run_subprocess, all_test_files)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # cd keras/fuzz
    # python run_fuzz.py ../data/demo.py openvino
    # python run_fuzz.py ../data/combined_sources_keras_test_41986.py openvino

    starttime = datetime.datetime.now()
    collected_test_cases_file = sys.argv[1]
    SUT = sys.argv[2]  # [tvm, openvino]
    run_all_test(collected_test_cases_file, SUT)
    endtime = datetime.datetime.now()
    print("Finish all, time consuming(min): ", (endtime - starttime).seconds/60)
