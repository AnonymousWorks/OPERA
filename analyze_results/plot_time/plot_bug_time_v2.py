from matplotlib import pyplot as plot
import math
import matplotlib

COLORs = {"Random": "#F08080",
      "OPERA": "#1e90ff"}

LS = {"Random": ":",
      "OPERA": "-"}


def get_accumulate_bug_num(bug_line_list, max_time):
    cumulative_bug_list = []
    cumulative_bug_num = 0
    for i in range(max_time):
        if i in bug_line_list:
            cumulative_bug_num += bug_line_list.count(i)
        cumulative_bug_list.append(cumulative_bug_num)
    return cumulative_bug_list


def plot_all(all_method_res_dict, sut, project, max_time):
    config = {
        "font.family": "sans-serif",  # 使用衬线体
        "font.sans-serif": ["Helvetica"],  # 全局默认使用衬线宋体,
        "font.size": 22,
        "axes.unicode_minus": False,
        # "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
    }
    matplotlib.rcParams['xtick.labelsize'] = 15
    plot.rcParams.update(config)
    fig, ax = plot.subplots(figsize=(7, 4))
    ax.set_ylabel("# Bugs", ) # fontweight='bold')
    # ax.locator_params(axis='x', nbins=7)  # the number of num in axis.
    # ax.locator_params(axis='y', nbins=12)  # the number of num in axis.
    ax.tick_params(axis='both', labelsize=20)
    for method, cumulative_bug_list in all_method_res_dict.items():

        bugs_id_list = list(range(len(cumulative_bug_list)))
        if project == "onnx":
            bugs_id_list_mins = list(range(math.ceil(len(cumulative_bug_list) / 60)))
            cumulative_bug_list_mins = [cumulative_bug_list[j * 60] for j in range(0, len(bugs_id_list_mins) - 1)]
            cumulative_bug_list_mins.append(cumulative_bug_list[-1])
            ax.plot(bugs_id_list_mins, cumulative_bug_list_mins,  label=method, linewidth=3, ls=LS[method], color=COLORs[method])
        else:
            bugs_id_list_hour = list(range(math.ceil(len(cumulative_bug_list) / 3600)))
            cumulative_bug_list_hour = [cumulative_bug_list[j * 3600] for j in range(0, len(bugs_id_list_hour) - 1)]
            cumulative_bug_list_hour.append(cumulative_bug_list[-1])
            ax.plot(bugs_id_list_hour, cumulative_bug_list_hour,  label=method, linewidth=3, ls=LS[method], color=COLORs[method])
        # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='plain', axis='x')
    if sut == 'tvm':
        if project == "torch":
            ax.set_xlabel("Time (h)", )  # fontweight='bold')
            ax.set_xlim([-1, 48])
            ax.set_ylim([-1, 35])
        elif project == "keras":
            ax.set_xlabel("Time (h)", )  # fontweight='bold')
            ax.set_xlim([-1, 80])
            ax.set_ylim([-1, 35])
        elif project == "onnx":
            ax.set_xlabel("Time (min)", )  # fontweight='bold')
            ax.set_xlim([-1, 35])
            ax.set_ylim([-1, 18])
    elif sut == 'ov':
        if project == "torch":
            ax.set_xlabel("Time (h)", )  # fontweight='bold')
            ax.set_xlim([-1, 40])
            ax.set_ylim([-1, 18])
        elif project == "keras":
            ax.set_xlabel("Time (h)", )  # fontweight='bold')
            ax.set_xlim([-1, 60])
            ax.set_ylim([-1, 12])
        elif project == "onnx":
            ax.set_xlabel("Time (min)", )  # fontweight='bold')
            ax.set_xlim([-1, 35])
            ax.set_ylim([-1, 25])

    plot.tight_layout()
    plot.legend(loc='lower right', fontsize=20)
    plot.savefig(f"trends_{project}.pdf")
    plot.show()


def load_time_record(time_file, total_test_num):
    test_time_list = []
    with open(time_file, 'r') as time_f:
        all_lines = time_f.readlines()
    test_id = 0
    accumulate_time = 0
    before_time = 0
    for line in all_lines:
        temp = line.strip().split('\t')
        test_id = temp[0]
        this_time = int(float(temp[1])) + accumulate_time
        if this_time < before_time:
            accumulate_time += before_time
        test_time_list.append(this_time)

        before_time = this_time
        # print(this_time)

    end_test_id = int(test_id)+1
    res_test_num = total_test_num - end_test_id
    if end_test_id < total_test_num:
        for rest_test_id in range(res_test_num):
            test_time_list.append(before_time + rest_test_id*2)

    return test_time_list


bug_position_tvm_dict = {
'torch_our':[6, 12, 17, 40, 84, 101, 102, 110, 125, 149, 154, 249, 277, 309, 450, 488, 567, 626, 675, 1086, 1150, 1279, 1657, 2598, 3639, 5388, 6229, 6399, 11345, 16521, 16871],
'torch_random':[42, 77, 131, 176, 317, 390, 481, 530, 674, 1028, 1231, 1432, 1604, 1927, 2388, 2816, 3231, 3698, 4302, 4971, 6068, 6830, 8101, 10751, 12795, 15666, 19098, 21606, 27053, 32796, 44733],

'keras_our':[11, 14, 15, 17, 19, 21, 33, 41, 42, 76, 77, 103, 121, 126, 138, 159, 168, 170, 183, 203, 248, 300, 321, 426, 431, 554, 749, 893, 955, 962, 1342, 2465, 3896],
'keras_random':[5, 15, 36, 62, 108, 136, 221, 308, 357, 440, 642, 737, 820, 965, 1247, 1412, 1742, 2086, 2294, 2584, 2844, 3430, 4082, 4654, 5710, 7035, 8010, 9670, 11424, 13984, 17037, 20948, 23067],

'onnx_our':[103, 154, 157, 171, 184, 240, 259, 295, 370, 417, 421, 428, 469],
'onnx_random':[30, 87, 136, 196, 251, 306, 388, 481, 568, 660, 769, 843, 928],
}

bug_position_ov_dict = {
'torch_our': [30, 42, 49, 54, 59, 60, 62, 66, 88, 117, 139, 156, 3010, 3411, 4585, 16231, 18615],
'torch_random': [50, 85, 159, 372, 954, 1418, 2878, 3750, 4288, 5545, 7566, 9117, 10342, 13200, 17394, 22242, 35865],

'keras_our': [37, 130, 791, 833, 1117, 1198, 1749, 3363, 19792, 27396, 29225],
'keras_random': [26, 196, 412, 692, 1136, 2602, 4638, 6339, 9867, 15656, 26276],

'onnx_our': [20, 21, 22, 23, 27, 28, 64, 81, 88, 131, 134, 135, 136, 137, 138, 140, 152, 194, 200, 240, 245, 387],
'onnx_random': [18, 25, 70, 88, 121, 146, 164, 186, 202, 219, 255, 281, 304, 334, 370, 409, 448, 506, 574, 627, 712, 860],
}

bug_position_trt_dict = {

}
if __name__ == '__main__':

    SUT_list = ['tvm']  # ov, trt
    for sut in SUT_list:
        bug_position_dict = eval(f"bug_position_{sut}_dict")
        projects = ['torch', 'keras', 'onnx']
        all_res_dict = {}
        for project in projects:
            if project == 'keras':
                test_num = 41986
                tcp_time = 18
            elif project == 'torch':
                test_num = 64756
                tcp_time = 86
            elif project == 'onnx':
                test_num = 1013
                tcp_time = 10

            bug_time_relation = load_time_record(f'time_record_{sut}_{project}.txt', test_num)
            bug_time_relation = [i+tcp_time for i in bug_time_relation]

            # bug_time_relation = [i/3600 for i in bug_time_relation]
            our_each_bug_time = [bug_time_relation[i] for i in bug_position_dict[f'{project}_our']]
            all_bug_found_total_time = our_each_bug_time[-1] / 3600
            print(f"Total time of OPERA for {project} is {all_bug_found_total_time} hours")

            random_each_bug_time = [bug_time_relation[int(i)] for i in bug_position_dict[f'{project}_random']]
            all_bug_found_total_time = random_each_bug_time[-1] / 3600
            print(f"Total time of Random for {project} is {all_bug_found_total_time} hours")
            max_time = bug_time_relation[-1]
            print(f"Total time for all {project} tests {max_time/3600}\n")
            # print(our_each_bug_time)
            # print(random_each_bug_time)
            all_res_dict['OPERA'] = get_accumulate_bug_num(our_each_bug_time, max_time)
            all_res_dict['Random'] = get_accumulate_bug_num(random_each_bug_time, max_time)
            # all_res_dict['OPERA$_{random}$'] = get_accumulate_bug_num(random_each_bug_time, max_time)
            plot_all(all_res_dict, sut, project, max_time=max_time)
