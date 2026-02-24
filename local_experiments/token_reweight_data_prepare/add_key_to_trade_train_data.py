import json
import os
import hashlib


def calculate_md5(input_string):
    """计算字符串的MD5哈希值"""
    md5_hash = hashlib.md5()  # 创建MD5哈希对象
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()


if __name__ == "__main__":
    data_dir = r'/Users/guoxing.lan/projects/datasets/LLM/sft/mixed_data'
    for batch_index in range(0, 3):
        with open(os.path.join(data_dir, f"trade_data_batch_{batch_index}.json"), 'r', encoding='utf8') as rf:
            list_train_data = json.load(rf)
        for i, temp_data in enumerate(list_train_data):
            long_str = ""
            for j, message in enumerate(temp_data['messages'][1:]):
                long_str += message['content']
            dedup_key = calculate_md5(long_str) + "_page_info"
            temp_data['dedup_key'] = dedup_key
            temp_data['span_weights'] = [{
                "start": 0,
                "end": len(temp_data['messages'][-1]['content']),
                "weight": 1
            }]
        with open(os.path.join(data_dir, f"trade_data_with_key_batch_{batch_index}.json"), 'w', encoding='utf8') as wf:
            json.dump(list_train_data, wf, ensure_ascii=False, indent=2)
