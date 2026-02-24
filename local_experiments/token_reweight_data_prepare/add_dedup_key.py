import hashlib
from datetime import datetime
import os
import json
def calculate_md5(input_string):
    """计算字符串的MD5哈希值"""
    md5_hash = hashlib.md5()  # 创建MD5哈希对象
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()


if __name__=="__main__":
    data_dir=r'/Users/guoxing.lan/projects/datasets/LLM/sft/mixed_data'
    with open(os.path.join(data_dir,'page_info_single_turn_hangqing_final_0127.json'),'r') as rf:
        list_train_data=json.load(rf)

    for i, temp_data in enumerate(list_train_data):
        if 'dedup_key' not in temp_data:
            long_str=""
            for j, message in enumerate(temp_data['messages'][1:]):
                long_str+=message['content']
            dedup_key=calculate_md5(long_str)+"_page_info"
            temp_data['dedup_key']=dedup_key
    with open(os.path.join(data_dir,'page_info_single_turn_hangqing_final_with_key_0127.json'),'w') as wf:
        json.dump(list_train_data,wf,ensure_ascii=False,indent=2)
