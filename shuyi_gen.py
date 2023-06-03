import sys
import os

import random
import numpy as np

P_RANGE = [0.05, 0.5, 0.7]
MODELS = ['empty', 'small', 'mid', 'big']
SPECS = ['simple', 'parallel']
DIFFICULTY = ['easy', 'medium', 'difficult']
SIZES = [35,35,35]


# responsible for writing the file
def write_vnnlib(X, spec_type, spec_path, Y_shape=6):
    with open(spec_path, "w") as f:
        f.write("\n")
        for i in range(int(X.shape[0] / 2)):
            f.write(f"(declare-const X_{i} Real)\n")
        if spec_type == SPECS[0]:
            for i in range(6):
                f.write(f"(declare-const Y_{i} Real)\n")
        if spec_type == SPECS[1]:
            f.write(f"(declare-const Y_0 Real)\n")
        f.write("\n; Input constraints:\n")

        for i in range(X.shape[0]):
            if i % 2 == 0:
                f.write(f"(assert (>= X_{int(i / 2)} {X[i]}))\n")
            else:
                f.write(f"(assert (<= X_{int((i - 1) / 2)} {X[i]}))\n")

        f.write("\n; Output constraints:\n")
        if spec_type == SPECS[0]:
            if spec_type == SPECS[0]:
                cannot_be_largest = 0
            if spec_type == 1:
                cannot_be_largest = Y_shape - 1
            for i in range(Y_shape):
                if not i == cannot_be_largest:
                    f.write(f"(assert (<= Y_{i} Y_{cannot_be_largest}))\n")

        if spec_type == SPECS[1]:
            f.write(f"(assert (<= Y_0 0))\n\n")


def add_range(X, spec_type, p_range):
    ret = np.empty(X.shape[0] * 2)
    if spec_type == SPECS[0]:
        for i in range(X.shape[0]):
            if 15 < i < 32:
                ret[i * 2] = X[i]
                ret[i * 2 + 1] = X[i] + p_range
            else:
                ret[i * 2] = X[i]
                ret[i * 2 + 1] = X[i]
    if spec_type == SPECS[1]:
        for i in range(X.shape[0]):
            if 15 < i < 32:
                ret[i * 2] = X[i]
                ret[i * 2 + 1] = X[i] + p_range
            elif 63 < i < 80:
                ret[i * 2] = X[i]
                ret[i * 2 + 1] = X[i] + p_range
            else:
                ret[i * 2] = X[i]
                ret[i * 2 + 1] = X[i]
    return ret


def parser(num):
    index = int(num % 10000)
    num = int(num / 10000)
    p_range = P_RANGE[num % 10]
    num = int(num / 10)
    spec = SPECS[num % 10]
    num = int(num / 10)
    model = MODELS[num % 10]
    return index, p_range, model, spec


def get_time(all_dic, index):
    for i in range(all_dic.shape[0]):
        if (all_dic[i][0] == index):
            return all_dic[i][1], all_dic[i][2]
    return -1, -1


def gene_spec():
    if not os.path.exists('vnnlib'):
        os.makedirs('vnnlib')
    vnn_dir_path = 'vnnlib'
    onnx_dir_path = 'onnx'
    csv_data = []
    total_num = 0

    size_ptr = 0
    for difficulty in DIFFICULTY:
        indexes = list(np.load(f'./src/pensieve/pensieve_resources/pen_{difficulty}.npy'))
        dic = np.load(f'./src/pensieve/pensieve_resources/pen_{difficulty}_dic.npy')
        chosen_index = random.sample(indexes, SIZES[size_ptr])
        size_ptr += 1
        for i in chosen_index:
            if i == 0:
                continue
            index, p_range, model, spec = parser(i)
            vnn_path = f'{vnn_dir_path}/pensieve_{spec}_{total_num}.vnnlib'
            onnx_path = onnx_dir_path + '/pensieve_' + model + '_' + spec + '.onnx'
            input_array = np.load(f'./src/pensieve/pensieve_resources/{model}_{spec}.npy')[index]
            write_vnnlib(add_range(input_array, spec, p_range), spec, vnn_path)
            total_num += 1
            ground_truth, timeout = get_time(dic, i)
            if timeout == -1:
                continue
            csv_data.append([onnx_path, vnn_path, int(timeout)])
    return csv_data


def main(random_seed):
    random.seed(random_seed)
    return gene_spec()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: generate_properties.py <random seed>")
        exit(1)

    random_seed = int(sys.argv[1])
    main(random_seed)
