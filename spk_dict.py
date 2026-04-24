import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='从 train.txt 提取说话人映射表')
    parser.add_argument('data_dir', help='包含 train.txt 的数据目录')
    parser.add_argument('-o', '--output', default='spk_mapping.json', help='输出 JSON 文件路径（默认: spk_mapping.json）')
    args = parser.parse_args()

    retrieval_path = Path(args.data_dir) / 'train.txt'
    if not retrieval_path.exists():
        print(f"错误: 找不到 {retrieval_path}")
        return

    spk_dict = {}
    seen = set()

    with open(retrieval_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            spk = line.split('/')[0]
            if spk not in seen:
                seen.add(spk)
                spk_dict[spk] = len(seen) - 1

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(spk_dict, f, ensure_ascii=False, indent=2)

    print(f"映射表已保存至 {args.output}")
    print(spk_dict)

if __name__ == '__main__':
    main()