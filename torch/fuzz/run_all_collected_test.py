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

torch_run_head = """import torch
from torch.nn import Module
from test_tvm_torch import verify_model
from numpy import inf
def tensor(x):
    return x

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


def gen_test_case_file(test_str, frame, test_id):
    save_dir = f"{frame}_tests_docter"    #todo: change it
    if frame == 'keras':
        test_str = keras_run_head + test_str
    elif frame == 'torch':
        test_str = torch_run_head + test_str
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.system(f"cp test_tvm_{frame}.py {save_dir}")
    save_test_file_path = os.path.join(save_dir, f'test_{test_id}.py')
    with open(save_test_file_path, 'w', encoding='utf-8') as test_f:
        test_f.write(test_str)
    return save_test_file_path


def load_test_from_file(test_file, begin_id=1):
    # support keras and pytorch now!
    with open(test_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    all_test_str_list = []

    this_test_case = ''
    cnt = begin_id  # TODO: check the id num
    for line in all_lines:
        if line.startswith("verify_model") or line.startswith("layer_test"):
            line = line.strip()[:-1]
            if line.endswith(','):
                line += f"count={cnt},)"
            else:
                line += f",count={cnt},)"
            this_test_case += line
            if this_test_case not in all_test_str_list:
                all_test_str_list.append(this_test_case)
                cnt += 1
            this_test_case = ''
        else:
            this_test_case += line
    return all_test_str_list


def run_all_test(test_file, frame='keras'):
    #if 'torch' in test_file and 'docter' in test_file:
    #    begin_id = 32379
    #elif 'keras' in test_file and 'docter' in test_file:
    #    begin_id = 20994
    #else:
    #    begin_id = 1
    #all_test_str_list = load_test_from_file(test_file, begin_id)
    #print(f"The collected test cases number in {frame} is : {len(all_test_str_list)}")
    #all_test_files = []
    #for cnt, test_str in enumerate(all_test_str_list):
    #    test_str = test_str.strip()
    #    save_test_file_path = gen_test_case_file(test_str, frame, cnt+begin_id)
    #    all_test_files.append(save_test_file_path)

    all_test_files = []
    test_dir = 'torch_tests_docter'
    for tc in os.listdir(test_dir):
        if 'test_tvm_torch.py' in tc:
            continue
        elif tc.endswith('.py'):
            tc_file = os.path.join(test_dir, tc)
            all_test_files.append(tc_file)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()//2)
    pool.map(run_subprocess, all_test_files)

    pool.close()
    pool.join()


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    # collected_test_cases_file = f"borrow_all_test.py"
    # collected_test_cases_file = f"borrow_all_test_deduplicate.py"
    frame = sys.argv[1]
    collected_test_cases_file = sys.argv[2]
    run_all_test(collected_test_cases_file, frame)

    endtime = datetime.datetime.now()
    print("Finish all, time consuming(min): ", (endtime - starttime).seconds/60)
