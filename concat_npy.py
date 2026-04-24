import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='拼接两个 .npy 文件')
    parser.add_argument('file1', help='第一个输入文件')
    parser.add_argument('file2', help='第二个输入文件')
    parser.add_argument('-o', '--output', default='lengths.npy', help='输出文件路径（默认 lengths.npy）')
    args = parser.parse_args()

    arr1 = np.load(args.file1)
    arr2 = np.load(args.file2)
    concatenated = np.concatenate((arr1, arr2))
    np.save(args.output, concatenated)
    print(f'已保存拼接结果至 {args.output}')

if __name__ == '__main__':
    main()