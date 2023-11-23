import numpy as np


def get_accumulate_bug_num(bug_line_list, test_case_num):
    cumulative_bug_list = []
    cumulative_bug_num = 0
    for i in range(test_case_num):
        if i in bug_line_list:
            cumulative_bug_num += bug_line_list.count(i)
        cumulative_bug_list.append(cumulative_bug_num)

    return cumulative_bug_list


def cal_apfd(test_num, bug_list):
    average_rank = 0
    bug_num = len(bug_list)
    # print(bug_num)
    for rank in bug_list:
        average_rank += rank
    APFD = 1 - average_rank/(bug_num*test_num) + 1/(2*test_num)
    return round(APFD, 4)


def calc_rauc(bug_list, test_num, percent):
    use_test_num = int(test_num * percent)

    ideal_data = range(len(bug_list))
    cumulative_ideal_data = get_accumulate_bug_num(ideal_data, use_test_num)
    cumulative_ideal_data = np.array(cumulative_ideal_data)
    res_ideal = np.trapz(cumulative_ideal_data)

    cumulative_bug_list = get_accumulate_bug_num(bug_list, use_test_num)
    cumulative_bug_list = np.array(cumulative_bug_list)
    res_this = np.trapz(cumulative_bug_list)

    return round(res_this / res_ideal, 4)


all_tvm_tcp_res = {
'torch_our':[6, 12, 17, 40, 84, 101, 102, 110, 125, 149, 154, 249, 277, 309, 450, 488, 567, 626, 675, 1086, 1150, 1279, 1657, 2598, 3639, 5388, 6229, 6399, 11345, 16521, 16871],
'torch_random':[42, 77, 131, 176, 317, 390, 481, 530, 674, 1028, 1231, 1432, 1604, 1927, 2388, 2816, 3231, 3698, 4302, 4971, 6068, 6830, 8101, 10751, 12795, 15666, 19098, 21606, 27053, 32796, 44733],
'torch_fast':[19, 23, 60, 67, 129, 136, 169, 421, 637, 645, 1238, 1552, 1608, 1679, 1770, 2045, 2215, 2974, 3033, 3782, 4051, 4131, 6392, 7134, 10643, 23634, 26969, 48966, 50548, 54339, 60713],
'torch_cov':[7793, 7829, 7910, 10046, 12511, 12992, 13899, 14052, 14418, 14863, 15446, 15514, 17839, 18435, 18762, 19020, 19031, 19398, 19889, 20868, 21011, 25001, 28166, 30482, 31475, 35765, 35907, 36248, 36383, 36477],
'torch_delta_cov':[46, 47, 85, 326, 427, 691, 977, 2190, 2902, 2917, 3093, 3497, 4296, 4385, 8389, 8846, 9200, 9214, 9719, 10371, 11707, 11895, 15898, 24831, 26153, 32003, 32312, 32454, 33043, 33300],

'keras_our':[11, 14, 15, 17, 19, 21, 33, 41, 42, 76, 77, 103, 121, 126, 138, 159, 168, 170, 183, 203, 248, 300, 321, 426, 431, 554, 749, 893, 955, 962, 1342, 2465, 3896],
'keras_random':[5, 15, 36, 62, 108, 136, 221, 308, 357, 440, 642, 737, 820, 965, 1247, 1412, 1742, 2086, 2294, 2584, 2844, 3430, 4082, 4654, 5710, 7035, 8010, 9670, 11424, 13984, 17037, 20948, 23067],
'keras_fast':[2, 14, 15, 15, 245, 263, 301, 381, 673, 694, 893, 906, 1159, 1220, 1224, 1314, 1799, 1833, 1958, 1975, 3815, 4253, 5670, 5848, 6394, 6481, 7247, 7605, 10983, 15497, 22731, 31505, 39672],
'keras_cov':[1, 3, 5, 6, 8, 12, 26, 27, 28, 33, 4570, 6995, 14654, 16179, 18111, 18114, 18229, 18240, 18376, 18884, 19773, 19789, 21010, 23307, 26457, 26492, 31335, 31470, 31602, 33551, 33597, 34797, 34798],
'keras_delta_cov':[2, 3, 6, 7, 10, 17, 21, 22, 30, 61, 65, 78, 83, 99, 111, 119, 120, 144, 150, 164, 170, 183, 203, 212, 217, 1554, 3287, 3316, 4010, 5086, 5274, 6989, 10014],

'onnx_our':[103, 154, 157, 171, 184, 240, 259, 295, 370, 417, 421, 428, 469],
'onnx_random':[30, 87, 136, 196, 251, 306, 388, 481, 568, 660, 769, 843, 928],
'onnx_fast':[11, 46, 59, 90, 320, 429, 649, 751, 758, 885, 928, 941, 946],
'onnx_cov':[7, 16, 53, 100, 145, 229, 298, 400, 407, 572, 615, 689, 1013],
'onnx_delta_cov':[17, 285, 305, 428, 501, 505, 531, 638, 681, 789, 823, 925, 935],
}
all_ov_tcp_res = {
'torch_our':[30, 42, 49, 54, 59, 60, 62, 66, 88, 117, 139, 156, 3010, 3411, 4585, 16231, 18615],
'torch_random':[50, 85, 159, 372, 954, 1418, 2878, 3750, 4288, 5545, 7566, 9117, 10342, 13200, 17394, 22242, 35865],
'torch_fast':[62, 67, 76, 100, 494, 1437, 2070, 2215, 2775, 2932, 3992, 4938, 6881, 12111, 12929, 29489, 48849],
'torch_cov':[2481, 7891, 7941, 10087, 12267, 15324, 15485, 19020, 21011, 22007, 26183, 33737, 37127],
'torch_delta_cov':[379, 537, 715, 2301, 4127, 4348, 9200, 11895, 13291, 19023, 29253, 32334, 59403],

'keras_our':[3, 4, 9, 245, 420, 535, 541, 1468, 3994, 12324, 12425],
'keras_random':[83, 144, 394, 740, 2112, 4907, 6253, 8511, 11492, 14827, 20153],
'keras_fast':[3, 125, 360, 985, 1007, 3111, 4400, 4765, 4905, 17305, 20212],
'keras_cov':[5, 2165, 4218, 6963, 7015, 7559, 13375, 13532, 14295, 26677, 31504],
'keras_delta_cov':[2, 30, 94, 204, 951, 1413, 2019, 11971, 20890, 21021, 24779],

'onnx_our':[20, 23, 64, 81, 88, 131, 140, 152, 194, 200, 240, 245, 387],
'onnx_random':[22, 40, 81, 128, 176, 228, 293, 381, 437, 506, 607, 761, 832],
'onnx_fast':[11, 19, 25, 53, 211, 292, 323, 391, 414, 606, 619, 751, 941],
'onnx_cov':[16, 147, 187, 229, 293, 298, 358, 370, 407, 411, 458, 717, 907],
'onnx_delta_cov':[107, 108, 276, 305, 533, 647, 675, 681, 778, 827, 859, 935, 982],

}

if __name__ == '__main__':
    num_keras = 41986
    num_torch = 64756
    num_onnx = 1013

    result_apfd_str = ''
    result_rauc_str = ''
    SUT_list = ['tvm', 'ov']  # , 'trt'
    for sut in SUT_list:
        print(f"Result for {sut}:")
        for k, v in eval(f"all_{sut}_tcp_res").items():
            # print(k)
            bug_list = [int(i) for i in v]
            project_method = k.split('_')
            project = project_method[0]
            method = project_method[1]
            test_num = eval(f'num_{project}')
            APDF = cal_apfd(test_num, bug_list)
            RAUC = calc_rauc(bug_list, test_num, 1)
            # print(project_method, APDF)
            result_apfd_str += str(APDF) + '\t'
            result_rauc_str += str(RAUC) + '\t'
            if method == 'delta':
                # print(result_apfd_str)
                print(result_rauc_str)
                result_apfd_str = ''
                result_rauc_str = ''
    print()
