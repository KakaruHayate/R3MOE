import json

def generate_and_save_spk_mapping(retrieval_path, save_path="spk_mapping.json"):
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

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(spk_dict, f, ensure_ascii=False, indent=2)
    
    return spk_dict

if __name__ == '__main__':
    mapping = generate_and_save_spk_mapping('E:\\GeneralCurveEstimator-SSL\\data0311\\train.txt')
    print(mapping)