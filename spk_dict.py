import json

def generate_and_save_spk_mapping(retrieval_path, save_path="spk_mapping.json"):
    spk_dict = {}
    seen = set()
    
    with open(retrieval_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 提取第一个路径元素
            spk = line.split('/')[0]
            
            # 维护有序且唯一的说话人列表
            if spk not in seen:
                seen.add(spk)
                spk_dict[spk] = len(seen) - 1  # ID从0开始

    # 保存为JSON文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(spk_dict, f, ensure_ascii=False, indent=2)
    
    return spk_dict

# 使用示例
if __name__ == '__main__':
    mapping = generate_and_save_spk_mapping('E:\\R3MOE-main\\data_0714\\train.txt')
    print("生成的映射表已保存为 spk_mapping.json")
    print(mapping)