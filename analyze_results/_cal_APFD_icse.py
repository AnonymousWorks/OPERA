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



all_tcp_res = {
'torch_our': [6, 12, 16, 38, 57, 84, 99, 107, 120, 141, 144, 192, 229, 254, 281, 408, 445, 503, 530, 628, 974, 1083, 1385, 2136, 2881, 4133, 4590, 4677, 5438, 14690, 15032],
'torch_random': [36.4, 61.3, 100.4, 143.6, 285.3, 348.8, 465.8, 560.0, 698.4, 886.7, 1251.6, 1630.8, 1872.8, 2293.2, 2970.3, 3391.2, 3818.4, 4536.5, 5622.2, 6364.9, 7292.2, 9240.5, 12048.9, 14634.9, 16913.6, 18815.1, 21389.3, 25391.8, 28016.8, 33482.0, 46731.3],
'torch_fast': [19, 23, 59, 66, 127, 133, 164, 398, 606, 613, 1168, 1465, 1518, 1585, 1670, 1929, 2089, 2789, 2842, 3528, 3782, 3859, 5972, 6654, 9894, 21999, 25216, 45630, 47079, 50617, 56512],
'torch_cov': [7793, 7827, 7906, 10021, 12467, 12947, 13849, 14000, 14365, 14806, 15385, 15453, 17763, 18353, 18678, 18934, 18945, 19310, 19799, 20765, 20907, 24857, 27998, 30290, 31276, 35518, 35628, 35918, 36033, 36112],
'torch_delta_cov': [46, 47, 85, 326, 427, 691, 977, 2185, 2893, 2908, 3082, 3483, 4275, 4364, 8342, 8797, 9148, 9162, 9665, 10314, 11635, 11822, 15784, 24645, 25956, 31743, 32014, 32138, 32658, 32887],



# 'keras_our': [3, 10, 13, 14, 15, 31, 32, 43, 61, 62, 86, 99, 103, 108, 118, 145, 156, 172, 184, 212, 252, 296, 344, 352, 363, 419, 585, 700, 750, 757, 5241, 10752, 11187],
'keras_our':   [4, 11, 14, 15, 16, 19, 21, 35, 36, 46, 48, 58, 65, 66, 67, 89, 100, 103, 108, 143, 177, 219, 260, 278, 293, 354, 594, 701, 814, 1125, 1134, 1533, 9390],
'keras_random':[8.1, 32.7, 75.1, 154.3, 191.7, 232.4, 267.5, 330.3, 417.2, 496.4, 587.3, 681.0, 777.5, 922.3, 1014.5, 1215.4, 1380.0, 1781.8, 1878.8, 2208.6, 2788.2, 3447.1, 3947.7, 4916.8, 6155.0, 7615.7, 9032.7, 12003.4, 14080.6, 16286.2, 18480.2, 23891.3, 29451.4],
'keras_fast': [2, 11, 12, 12, 191, 208, 238, 304, 539, 557, 699, 707, 898, 948, 951, 1028, 1418, 1445, 1551, 1567, 3021, 3379, 4490, 4631, 5064, 5135, 5746, 6033, 8714, 12291, 17961, 24943, 31397],
'keras_cov': [1, 2, 3, 4, 5, 6, 17, 18, 19, 19, 2820, 4362, 7453, 8346, 9771, 9774, 9789, 9796, 9928, 10404, 11242, 11258, 12465, 14661, 17777, 17807, 22572, 22694, 22823, 24695, 24738, 25927, 25928],
'keras_delta_cov': [2, 3, 6, 7, 8, 15, 19, 20, 27, 56, 59, 70, 75, 87, 98, 105, 106, 129, 134, 143, 149, 162, 178, 182, 186, 1496, 3183, 3212, 3892, 4942, 5123, 6800, 9761],



'onnx_our': [103, 154, 157, 169, 181, 237, 256, 291, 333, 364, 408, 412, 419, 458],
'onnx_random': [25.7, 87.7, 129.7, 181.7, 240.2, 302.8, 355.6, 447.8, 532.7, 616.3, 700.2, 751.8, 824.0, 933.0],
'onnx_fast': [11, 46, 59, 62, 88, 314, 422, 640, 739, 745, 871, 914, 927, 932],
'onnx_cov': [7, 16, 52, 98, 143, 227, 295, 394, 401, 526, 566, 608, 682, 997],
'onnx_delta_cov': [17, 285, 305, 322, 426, 499, 502, 528, 634, 677, 774, 808, 920],

}

if __name__ == '__main__':
    num_keras = 41986
    num_torch = 64756
    num_onnx = 1013

    result_apfd_str = ''
    result_rauc_str = ''
    for k, v in all_tcp_res.items():
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
            print(result_apfd_str)
            # print(result_rauc_str)
            result_apfd_str = ''
            result_rauc_str = ''
