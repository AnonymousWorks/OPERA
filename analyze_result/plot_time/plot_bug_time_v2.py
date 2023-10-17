from matplotlib import pyplot as plot
import math
from matplotlib import rc
# rc('text', usetex=True)


# COLORs = {"Random": "#AF58BA",
#       "OPERA": "#FFC325"}

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


def plot_all(all_method_res_dict, project, max_time):
    config = {
        "font.family": "sans-serif",  # 使用衬线体
        "font.sans-serif": ["Helvetica"],  # 全局默认使用衬线宋体,
        "font.size": 22,
        "axes.unicode_minus": False,
        # "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
    }

    import matplotlib
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


if __name__ == '__main__':
    bug_position_dict = {
    # 'keras_our': [3, 10, 13, 14, 15, 31, 32, 43, 61, 62, 86, 99, 103, 108, 118, 145, 156, 172, 184, 212, 252, 296, 344,
                  # 352, 363, 419, 585, 700, 750, 757, 5241, 10752, 11187],
    'keras_our': [4, 11, 14, 15, 16, 19, 21, 35, 36, 46, 48, 58, 65, 66, 67, 89, 100, 103, 108, 143, 177, 219, 260, 278,
                  293, 354, 594, 701, 814, 1125, 1134, 1533, 9390],
    'keras_random': [8.1, 32.7, 75.1, 154.3, 191.7, 232.4, 267.5, 330.3, 417.2, 496.4, 587.3, 681.0, 777.5, 922.3,
                     1014.5, 1215.4, 1380.0, 1781.8, 1878.8, 2208.6, 2788.2, 3447.1, 3947.7, 4916.8, 6155.0, 7615.7,
                     9032.7, 12003.4, 14080.6, 16286.2, 18480.2, 23891.3, 29451.4],

    'torch_our': [6, 12, 16, 38, 57, 84, 99, 107, 120, 141, 144, 192, 229, 254, 281, 408, 445, 503, 530, 628, 974,
                  1083, 1385, 2136, 2881, 4133, 4590, 4677, 5438, 14690, 15032],
    'torch_random': [36.4, 61.3, 100.4, 143.6, 285.3, 348.8, 465.8, 560.0, 698.4, 886.7, 1251.6, 1630.8, 1872.8,
                     2293.2, 2970.3, 3391.2, 3818.4, 4536.5, 5622.2, 6364.9, 7292.2, 9240.5, 12048.9, 14634.9,
                     16913.6, 18815.1, 21389.3, 25391.8, 28016.8, 33482.0, 46731.3],

    'onnx_our': [103, 154, 157, 169, 181, 237, 256, 291, 333, 364, 408, 412, 419, 458],
    'onnx_random': [25.7, 87.7, 129.7, 181.7, 240.2, 302.8, 355.6, 447.8, 532.7, 616.3, 700.2, 751.8, 824.0, 933.0],
    'onnx_NNSmith': [0, 120, 180, 420, 780, 1020, 1200, 1920, 4200, 7200, 19620, 93600]   #[2, 3, 14, 28, 30, 61, 583, 37, 1]

    }

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

        bug_time_relation = load_time_record(f'{project}_time_record.txt', test_num)
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
        plot_all(all_res_dict, project, max_time=max_time)

