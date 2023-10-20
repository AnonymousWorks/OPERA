import sys
import subprocess
import multiprocessing
import datetime
import os
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

onnx_run_head = """import onnx
from test_tvm_onnx import make_graph
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
    save_dir = f"{frame}_tests"
    if frame == 'keras':
        test_str = keras_run_head + test_str
    elif frame == 'torch':
        test_str = torch_run_head + test_str
    elif frame == 'onnx':
        test_str = onnx_run_head + test_str
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.system(f"cp test_tvm_{frame}.py {save_dir}")
    save_test_file_path = os.path.join(save_dir, f'test_{test_id}.py')
    with open(save_test_file_path, 'w', encoding='utf-8') as test_f:
        test_f.write(test_str)
    return save_test_file_path


def load_test_from_file(test_file):
    with open(test_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    all_test_str_list = []

    this_test_case = ''
    cnt = 0
    for line in all_lines:
        if line.startswith("verify_model") or line.startswith("layer_test") or line.startswith("make_graph"):
            line = line.strip()[:-1]
            if line.endswith(','):
                line += f"count={cnt},)"
            else:
                line += f",count={cnt},)"
            if "make_graph" in line:
                if "to be add" in line:
                    pass
            this_test_case += line
            if this_test_case not in all_test_str_list:
                cnt += 1
                print(cnt)
                all_test_str_list.append(this_test_case)
            this_test_case = ''
        else:
            this_test_case += line
    return all_test_str_list


def run_all_test(test_file, frame='keras'):
    all_test_str_list = load_test_from_file(test_file)
    print(f"The collected test cases number in {frame} is : {len(all_test_str_list)}")
    all_test_files = []
    for cnt, test_str in enumerate(all_test_str_list):
        test_str = test_str.strip()
        save_test_file_path = gen_test_case_file(test_str, frame, cnt)
        all_test_files.append(save_test_file_path)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()//3)
    pool.map(run_subprocess, all_test_files)

    pool.close()
    pool.join()


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    frame = sys.argv[1]
    collected_test_cases_file = sys.argv[2]
    run_all_test(collected_test_cases_file, frame)

    endtime = datetime.datetime.now()
    print("Finish all, time consuming(min): ", (endtime - starttime).seconds/60)

