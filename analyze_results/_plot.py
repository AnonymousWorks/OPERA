from matplotlib import pyplot as plot
import re


def get_accumulate_bug_num(bug_line_list, test_case_num):
    cumulative_bug_list = []
    cumulative_bug_num = 0
    for i in range(test_case_num):
        if i in bug_line_list:
            cumulative_bug_num += bug_line_list.count(i)
        cumulative_bug_list.append(cumulative_bug_num)

    return cumulative_bug_list


def my_plot(bug_line_list, test_case_num=1000):
    cumulative_bug_list = get_accumulate_bug_num(bug_line_list, test_case_num)
    plot.xlabel("# Test cases")
    plot.ylabel("# Bugs")
    plot.plot(range(len(cumulative_bug_list)), cumulative_bug_list)
    plot.show()


def plot_all(all_method_res_dict):
    plot.xlabel("# Test cases")
    plot.ylabel("# Bugs")
    for method, cumulative_bug_list in all_method_res_dict.items():
        plot.plot(range(len(cumulative_bug_list)), cumulative_bug_list, label=method)
    plot.legend()
    plot.show()


def preprocess_op_name(op_name):
    op_name = list(op_name)
    if op_name[0].islower():
        op_name[0] = op_name[0].upper()
    for i, ch in enumerate(op_name):
        if ch == '_':
            op_name[i+1] = op_name[i+1].upper()
    op_name = ''.join(op_name)
    op_name = op_name.replace('1d', '2d').replace('3d', '2d').replace('1D', '2D').replace('3D', '2D').replace('_', '')\
        .replace('Celu', 'CELU').replace('MseLoss', 'MSELoss').replace('LpPool', 'LPPool').replace('Elu', 'ELU')
        # .replace('2d', '')

    return op_name


def run_parse(bugs_file):
    with open(bugs_file, 'r', encoding='utf-8') as bug_f:
        all_bugs = bug_f.readlines()
    all_bugs_line_list = []
    all_bugs_key_list = set()
    all_bugs = sorted(all_bugs, key=lambda x: int(x.split("\t")[0]))
    bug_line_dict = {}  # k=line,v=bug_info

    for line in all_bugs:
        bug_line_id, bug_type, op, bug_line = line.split('\t')
        bug_line_id = str(int(bug_line_id))
        if op != 'Conv3DTranspose':  # 'Conv3DTranspose' and 'Conv2DTranspose' are two different bug according to patch.
            op = preprocess_op_name(op)  # rename the op_layer name
        bug_key = f'{op},{bug_line}'.strip()
        bug_info = bug_line.split('_')[-1].strip()
        # bug_key = re.sub(r'\d+', '__num__', bug_key)
        # for onnx
        # print(bug_info)
        if 'wrong results' == bug_type:
            bug_key = f'{op},wrong results'.strip()
            if op in ['RandomUniformLike', 'Bernoulli']:  # False positive
                continue
        if 'unknown type object' in bug_key:
            continue
        if "got multiple values for argument 'axis'" in bug_key and 'Reduce' in op:
            bug_key = "Reduce, TypeError_python/tvm/relay/frontend/common.py_453_sum got multiple values for argument 'axis'"
        # end for onnx

        # fp in pytorch
        if 'avoid ambiguity in the code' in bug_key:
            continue
        # end for pytorch
        if 'The size must exactly match' in bug_key:  # in graph, same with "is expected to be float32"
            bug_key = "is expected to be float32"
        if "is expected to be float32" in bug_key:  # in vm model, same with 'size must exactly match'
            bug_key = "is expected to be float32"
        elif 'Do not know how to handle type' in bug_key:
            bug_key = 'Do not know how to handle type'
        elif 'Cropping2D' == op and 'is invalid for axis=' in bug_info:
            continue  # this is a bug in keras. new keras version fixed it.
        elif 't argument must be a string, a bytes-like object or a number,' in bug_info:
            continue  # this is a false positive, 'return_runtime' attribute is not a valid attribute. it just for testing.

        elif "exceeds maximum of" in bug_info or "quant param not found" in bug_info or 'make uint from negative value' in bug_info:
            continue  # false positive
        elif 'Divide by zero' == bug_info and op in ['InstanceNorm', 'InstanceNorm2d', 'AdaptiveMaxPool2d']:
            continue  # a workaround to fileout the input_shape includes 0
        elif 'division by zero' == bug_info and op == 'Normalize':
            continue
        elif 'is expected to be int64' in bug_info:
            continue
        # ---------------------
        elif 'not support' in bug_info or 'only supported' in bug_info or 'Support only' in bug_info \
                or "not handled yet" in bug_info:
            continue
        elif "<class 'NoneType'>" in bug_info:
            bug_key = "<class 'NoneType'>"
        bug_line_dict[f"{bug_line_id}"] = bug_key
        if bug_key not in all_bugs_key_list:
            all_bugs_key_list.add(bug_key)
            all_bugs_line_list.append(int(bug_line_id))
            if 'is not valid for operator Dense' in bug_info:  # one test case, 2 bugs.
                all_bugs_line_list.append(int(bug_line_id))
            print(bug_line_id, bug_key)
    # my_plot(all_bugs_line_list, test_case_num=20000)
    print(len(all_bugs_line_list))
    return all_bugs_line_list, all_bugs_key_list, bug_line_dict


def get_random_tcp_res(bug_line_dict, test_num, repeat_time=10):
    import random
    final_average_unique_bugs_line_list = []
    ranked_unique_bugs_line_list = []
    ranked_unique_bugs_key_set = set()
    for i in range(repeat_time):
        this_rank = list(range(1, test_num + 1))
        if i != 0:
            random.shuffle(this_rank)
        this_rank = [str(i) for i in this_rank]
        ranked_id = 1
        for test_id in this_rank:
            if test_id in bug_line_dict.keys():
                bug_info = bug_line_dict[test_id]
                if bug_info not in ranked_unique_bugs_key_set:
                    ranked_unique_bugs_key_set.add(bug_info)
                    ranked_unique_bugs_line_list.append(int(ranked_id))
                    if 'is not valid for operator Dense' in bug_info:  # one test case, 2 bugs.
                        ranked_unique_bugs_key_set.add(bug_info + "_2")
                        ranked_unique_bugs_line_list.append(int(ranked_id) + 1)
                else:
                    continue
            ranked_id += 1
        final_average_unique_bugs_line_list.append(ranked_unique_bugs_line_list)
        ranked_unique_bugs_line_list = []
        ranked_unique_bugs_key_set = set()

    import numpy as np
    final_average_unique_bugs_line_list = np.array(final_average_unique_bugs_line_list)
    final_average_unique_bugs_line_list = np.average(final_average_unique_bugs_line_list, 0)
    final_average_unique_bugs_line_list = list(final_average_unique_bugs_line_list)

    return final_average_unique_bugs_line_list


def get_ranked_tc_tcp_res(bug_line_dict, ranked_bugs_file):
    ranked_unique_bugs_line_list = []
    ranked_unique_bugs_key_set = set()
    with open(ranked_bugs_file, 'r') as r_f:
        all_lines = r_f.readlines()

    ranked_id = 1
    for line in all_lines:
        if line.startswith("layer_test") or line.startswith("verify_model") or line.startswith('make_graph'):
            test_id = line.strip().split("count=")[-1][:-2]
            # print(test_id)
            if test_id in bug_line_dict.keys():
                bug_info = bug_line_dict[test_id]
                if bug_info not in ranked_unique_bugs_key_set:
                    ranked_unique_bugs_key_set.add(bug_info)
                    ranked_unique_bugs_line_list.append(int(ranked_id))
                    if 'is not valid for operator Dense' in bug_info:  # one test case, 2 bugs.
                        ranked_unique_bugs_key_set.add(bug_info+"_2")
                        ranked_unique_bugs_line_list.append(int(ranked_id)+1)
                else:
                    continue
            ranked_id += 1
    return ranked_unique_bugs_line_list, ranked_unique_bugs_key_set


def run_keras():
    all_bugs_file = f'./keras/combined_sources_keras_bugs_41986.txt'
    ranked_bugs_file = f"keras/ranked_test_case_keras_new_final.py"
    # ranked_bugs_file = f"./keras/882ranked_test_case_keras_rrrr.py"
    # baseline tcp results
    baseline_line_tcp_file = f"keras/line.py"
    baseline_delta_line_tcp_file = f"keras/delta_line.py"
    baseline_branch_tcp_file = f"keras/branch.py"
    baseline_delta_branch_tcp_file = f"keras/delta_branch.py"
    baseline_fast_tcp_file = f"keras/fast.py"

    test_case_num = 41986  # 41986
    all_methods_accumulate_bugs_dict = {}

    print("all bugs:...\n")
    all_bugs_line_list, all_bugs_list, all_bug_line_dict = run_parse(all_bugs_file)

    # my_plot(all_bugs_line_list, test_case_num=20000)
    cumulative_bug_list = get_accumulate_bug_num(all_bugs_line_list, test_case_num)
    all_methods_accumulate_bugs_dict['random'] = cumulative_bug_list

    print("test cases number which can detect a bug: ", len(all_bug_line_dict))

    print("\n\nranked tc bugs:...\n")
    # our_tcp_bugs_line_list, our_tcp_bugs_list, our_tcp_bug_line_dict = run_parse(ranked_bugs_file)
    # my_plot(our_tcp_bugs_line_list, test_case_num=20000)
    our_tcp_bugs_line_list, our_tcp_bugs_list = get_ranked_tc_tcp_res(all_bug_line_dict, ranked_bugs_file)
    print("'keras_our':", our_tcp_bugs_line_list)
    cumulative_bug_list = get_accumulate_bug_num(our_tcp_bugs_line_list, test_case_num)
    all_methods_accumulate_bugs_dict['our'] = cumulative_bug_list

    # -------------------------------baseline------------------------------------------------------------
    average_random_unique_bugs_line_list = get_random_tcp_res(all_bug_line_dict, test_case_num)
    print("'keras_random':", average_random_unique_bugs_line_list)
    # baseline: line

    ranked_unique_bugs_line_list, ranked_unique_bugs_key_set = get_ranked_tc_tcp_res(all_bug_line_dict,
                                                                                     baseline_line_tcp_file)
    # print(ranked_unique_bugs_line_list)
    # my_plot(ranked_unique_bugs_line_list, test_case_num=20000)
    cumulative_bug_list = get_accumulate_bug_num(ranked_unique_bugs_line_list, test_case_num)
    print("'keras_cov':", ranked_unique_bugs_line_list)
    all_methods_accumulate_bugs_dict['line_cov'] = cumulative_bug_list

    # baseline: line_delta_cov
    ranked_unique_bugs_line_list, _ = get_ranked_tc_tcp_res(all_bug_line_dict, baseline_delta_line_tcp_file)
    cumulative_bug_list = get_accumulate_bug_num(ranked_unique_bugs_line_list, test_case_num)
    all_methods_accumulate_bugs_dict['delta_line_cov'] = cumulative_bug_list
    print("'keras_delta_cov':",ranked_unique_bugs_line_list)
    #
    # # baseline: branch_delta_cov
    # ranked_unique_bugs_line_list, _ = get_ranked_tc_tcp_res(all_bug_line_dict, baseline_branch_tcp_file)
    # cumulative_bug_list = get_accumulate_bug_num(ranked_unique_bugs_line_list, test_case_num)
    # all_methods_accumulate_bugs_dict['branch_cov'] = cumulative_bug_list
    #
    # # baseline: branch_delta_cov
    # ranked_unique_bugs_line_list, _ = get_ranked_tc_tcp_res(all_bug_line_dict, baseline_delta_branch_tcp_file)
    # cumulative_bug_list = get_accumulate_bug_num(ranked_unique_bugs_line_list, test_case_num)
    # all_methods_accumulate_bugs_dict['delta_branch_cov'] = cumulative_bug_list

    # baseline: fast_tcp
    ranked_unique_bugs_line_list, _ = get_ranked_tc_tcp_res(all_bug_line_dict, baseline_fast_tcp_file)
    cumulative_bug_list = get_accumulate_bug_num(ranked_unique_bugs_line_list, test_case_num)
    all_methods_accumulate_bugs_dict['FAST'] = cumulative_bug_list
    print("'keras_fast':",ranked_unique_bugs_line_list)
    plot_all(all_methods_accumulate_bugs_dict)


    # our tcp results.
    undetected_bugs = set(all_bugs_list) - set(our_tcp_bugs_list)
    print("\n\nundetected bugs:\n")
    for bug in undetected_bugs:
        print(bug)
    new_bugs = set(our_tcp_bugs_list) - set(all_bugs_list)
    print('\n\nnew bugs:\n')
    for bug in new_bugs:
        print(bug)


def run_torch():
    # all_bugs_file = f"./detected_bugs/torch/detected_bugs_torch_new.txt"
    all_bugs_file = f"./torch/combined_sources_torch_bugs_64756.txt"
    ranked_bugs_file = f"torch/torch_ranked_test_case_new_final.py"

    # baseline tcp results
    baseline_line_tcp_file = f"torch/line.py"
    baseline_delta_line_tcp_file = f"torch/delta_line.py"
    baseline_branch_tcp_file = f"torch/branch.py"
    baseline_delta_branch_tcp_file = f"torch/delta_branch.py"
    baseline_fast_tcp_file = f"torch/torch_fast_final.py"

    test_case_num = 64756
    all_methods_accumulate_bugs_dict = {}

    print("all bugs:...\n")
    all_bugs_line_list, all_bugs_list, all_bug_line_dict = run_parse(all_bugs_file)

    cumulative_bug_list = get_accumulate_bug_num(all_bugs_line_list, test_case_num)
    all_methods_accumulate_bugs_dict['random'] = cumulative_bug_list

    # _, our_tcp_bugs_list,_ = run_parse("../detected_bugs/detected_bugs_torch_docter_all.txt")

    print("test cases number which can detect a bug: ", len(all_bug_line_dict))
    print('---------------------------------each tcp method--------------------------------------------')
    print("\n\nranked tc bugs:...\n")
    # our_tcp_bugs_line_list, our_tcp_bugs_list, our_tcp_bug_line_dict = run_parse(ranked_bugs_file)
    our_tcp_bugs_line_list, our_tcp_bugs_list = get_ranked_tc_tcp_res(all_bug_line_dict, ranked_bugs_file)
    print("'torch_our':", our_tcp_bugs_line_list)
    # my_plot(our_tcp_bugs_line_list, test_case_num=20000)
    cumulative_bug_list = get_accumulate_bug_num(our_tcp_bugs_line_list, test_case_num)
    all_methods_accumulate_bugs_dict['our'] = cumulative_bug_list

    # -------------------------------baseline------------------------------------------------------------
    average_random_unique_bugs_line_list = get_random_tcp_res(all_bug_line_dict, test_case_num)
    print("'torch_random':", average_random_unique_bugs_line_list)
    # baseline: line
    ranked_unique_bugs_line_list, ranked_unique_bugs_key_set = get_ranked_tc_tcp_res(all_bug_line_dict,
                                                                                     baseline_line_tcp_file)
    print("'torch_cov':",ranked_unique_bugs_line_list)
    # print(ranked_unique_bugs_line_list)
    # my_plot(ranked_unique_bugs_line_list, test_case_num=20000)
    cumulative_bug_list = get_accumulate_bug_num(ranked_unique_bugs_line_list, test_case_num)
    all_methods_accumulate_bugs_dict['line_cov'] = cumulative_bug_list

    # baseline: line_delta_cov
    ranked_unique_bugs_line_list, _ = get_ranked_tc_tcp_res(all_bug_line_dict, baseline_delta_line_tcp_file)
    print("'torch_delta_cov':",ranked_unique_bugs_line_list)
    cumulative_bug_list = get_accumulate_bug_num(ranked_unique_bugs_line_list, test_case_num)
    all_methods_accumulate_bugs_dict['delta_line_cov'] = cumulative_bug_list

    # # baseline: branch_cov
    # ranked_unique_bugs_line_list, _ = get_ranked_tc_tcp_res(all_bug_line_dict, baseline_branch_tcp_file)
    # cumulative_bug_list = get_accumulate_bug_num(ranked_unique_bugs_line_list, test_case_num)
    # all_methods_accumulate_bugs_dict['branch_cov'] = cumulative_bug_list
    #
    # # baseline: branch_delta_cov
    # ranked_unique_bugs_line_list, _ = get_ranked_tc_tcp_res(all_bug_line_dict, baseline_delta_branch_tcp_file)
    # cumulative_bug_list = get_accumulate_bug_num(ranked_unique_bugs_line_list, test_case_num)
    # all_methods_accumulate_bugs_dict['delta_branch_cov'] = cumulative_bug_list

    # baseline: fast_tcp
    ranked_unique_bugs_line_list, _ = get_ranked_tc_tcp_res(all_bug_line_dict, baseline_fast_tcp_file)
    print("'torch_fast':",ranked_unique_bugs_line_list)
    cumulative_bug_list = get_accumulate_bug_num(ranked_unique_bugs_line_list, test_case_num)
    all_methods_accumulate_bugs_dict['FAST'] = cumulative_bug_list

    plot_all(all_methods_accumulate_bugs_dict)


    # our tcp results.
    undetected_bugs = set(all_bugs_list) - set(our_tcp_bugs_list)
    print("\n\nundetected bugs:\n")
    for bug in undetected_bugs:
        print(bug)
    new_bugs = set(our_tcp_bugs_list) - set(all_bugs_list)
    print('\n\nnew bugs:\n')
    for bug in new_bugs:
        print(bug)


def run_onnx():
    all_bugs_file = f"./onnx/combined_sources_onnx_bugs_1013.txt"
    ranked_bugs_file = "onnx/onnx_ranked_test_case_new_final.py"

    # baseline tcp results
    baseline_line_tcp_file = f"onnx/line.py"
    baseline_delta_line_tcp_file = f"onnx/delta_line.py"
    baseline_branch_tcp_file = f"onnx/branch.py"
    baseline_delta_branch_tcp_file = f"onnx/delta_branch.py"
    baseline_fast_tcp_file = f"onnx/fast_onnx.py"

    test_case_num = 1013
    all_methods_accumulate_bugs_dict = {}

    print("all bugs:...\n")
    all_bugs_line_list, all_bugs_list, all_bug_line_dict = run_parse(all_bugs_file)
    # my_plot(all_bugs_line_list, test_case_num=20000)
    cumulative_bug_list = get_accumulate_bug_num(all_bugs_line_list, test_case_num)
    all_methods_accumulate_bugs_dict['random'] = cumulative_bug_list

    print("test cases number which can detect a bug: ", len(all_bug_line_dict))
    print('---------------------------------each tcp method--------------------------------------------')
    print("\n\nranked tc bugs:...\n")
    # our_tcp_bugs_line_list, our_tcp_bugs_list, our_tcp_bug_line_dict = run_parse(ranked_bugs_file)
    our_tcp_bugs_line_list, our_tcp_bugs_list = get_ranked_tc_tcp_res(all_bug_line_dict, ranked_bugs_file)
    print("'onnx_our':",our_tcp_bugs_line_list)
    # my_plot(our_tcp_bugs_line_list, test_case_num=20000)
    cumulative_bug_list = get_accumulate_bug_num(our_tcp_bugs_line_list, test_case_num)
    all_methods_accumulate_bugs_dict['our'] = cumulative_bug_list
    # for i in our_tcp_bugs_list:
    #     print(i)

    # -------------------------------baseline------------------------------------------------------------
    # baseline: random
    average_random_unique_bugs_line_list = get_random_tcp_res(all_bug_line_dict, test_case_num)
    print("'onnx_random':", average_random_unique_bugs_line_list)
    # baseline: line
    ranked_unique_bugs_line_list, ranked_unique_bugs_key_set = get_ranked_tc_tcp_res(all_bug_line_dict,
                                                                                     baseline_line_tcp_file)
    print("'onnx_cov':",ranked_unique_bugs_line_list)
    # my_plot(ranked_unique_bugs_line_list, test_case_num=20000)
    cumulative_bug_list = get_accumulate_bug_num(ranked_unique_bugs_line_list, test_case_num)
    all_methods_accumulate_bugs_dict['line_cov'] = cumulative_bug_list

    # baseline: line_delta_cov
    ranked_unique_bugs_line_list, _ = get_ranked_tc_tcp_res(all_bug_line_dict, baseline_delta_line_tcp_file)
    print("'onnx_delta_cov':",ranked_unique_bugs_line_list)
    cumulative_bug_list = get_accumulate_bug_num(ranked_unique_bugs_line_list, test_case_num)
    all_methods_accumulate_bugs_dict['delta_line_cov'] = cumulative_bug_list

    # # baseline: branch_delta_cov
    # ranked_unique_bugs_line_list, _ = get_ranked_tc_tcp_res(all_bug_line_dict, baseline_branch_tcp_file)
    # cumulative_bug_list = get_accumulate_bug_num(ranked_unique_bugs_line_list, test_case_num)
    # all_methods_accumulate_bugs_dict['branch_cov'] = cumulative_bug_list
    #
    # # baseline: branch_delta_cov
    # ranked_unique_bugs_line_list, _ = get_ranked_tc_tcp_res(all_bug_line_dict, baseline_delta_branch_tcp_file)
    # cumulative_bug_list = get_accumulate_bug_num(ranked_unique_bugs_line_list, test_case_num)
    # all_methods_accumulate_bugs_dict['delta_branch_cov'] = cumulative_bug_list

    # baseline: fast_tcp
    ranked_unique_bugs_line_list, _ = get_ranked_tc_tcp_res(all_bug_line_dict, baseline_fast_tcp_file)
    print("'onnx_fast':",ranked_unique_bugs_line_list)
    cumulative_bug_list = get_accumulate_bug_num(ranked_unique_bugs_line_list, test_case_num)
    all_methods_accumulate_bugs_dict['FAST'] = cumulative_bug_list

    plot_all(all_methods_accumulate_bugs_dict)

    # our tcp results.
    undetected_bugs = set(all_bugs_list) - set(our_tcp_bugs_list)
    print("\n\nundetected bugs:\n")
    for bug in undetected_bugs:
        print(bug)
    new_bugs = set(our_tcp_bugs_list) - set(all_bugs_list)
    print('\n\nnew bugs:\n')
    for bug in new_bugs:
        print(bug)


if __name__ == '__main__':
    run_keras()
    # run_torch()
    # run_onnx()
