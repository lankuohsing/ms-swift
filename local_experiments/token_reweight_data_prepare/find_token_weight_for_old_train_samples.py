import json
import os
from collections import defaultdict
if __name__=="__main__":
    dict_content_samples={}
    data_dir=r'/Users/guoxing.lan/projects/datasets/LLM/sft/mixed_data'
    with open(os.path.join(data_dir,"page_info_single_turn_hangqing_final_with_key_0127_token_reweight.json"),'r',encoding='utf8') as rf:
        list_data_with_token_reweight=json.load(rf)
    for i, temp_data in enumerate(list_data_with_token_reweight):
        long_content= ""
        for j, message in enumerate(temp_data['messages']):
            long_content+=message['content']
        dict_content_samples[long_content]=temp_data

    for batch_id in range(0,3):
        list_new_batch_data=[]
        with open(os.path.join(data_dir,f"page_trade_data_batch_{batch_id}.json"),'r',encoding='utf8') as rf:
            list_old_batch_data=json.load(rf)
        for i, temp_data in enumerate(list_old_batch_data):
            long_content= ""
            for j, message in enumerate(temp_data['messages']):
                long_content+=message['content']
            temp_sample=dict_content_samples[long_content]
            list_new_batch_data.append(temp_sample)
        with open(os.path.join(data_dir,f"page_trade_data_token_reweight_batch_{batch_id}.json"),'w',encoding='utf8') as wf:
            json.dump(list_new_batch_data, wf,ensure_ascii=False, indent=2)

