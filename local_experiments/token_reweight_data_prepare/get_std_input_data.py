import json
import os
import re


def extract_clauses(text):
    # 使用正则表达式匹配子句
    # 匹配模式：<子句编号X>后面的内容，直到下一个<子句编号Y>或文本结束
    pattern = r'<子句编号\d+>(.*?)(?=<子句编号\d+>|$)'
    clauses = re.findall(pattern, text, re.DOTALL)
    return clauses


if __name__ == "__main__":
    data_dir = r'/Users/guoxing.lan/projects/datasets/LLM/sft/mixed_data'
    list_data_judged_by_r1 = []
    span_weights = []
    list_train_data = []
    empty_reweight_count=0
    correct_num=0
    with open(os.path.join(data_dir, 'page_info_single_turn_hangqing_final_with_key_0127_r1_critical_tokens.jsonl'), 'r') as rf:
        for line in rf:
            temp_data = json.loads(line.strip())
            ground_truth_str = temp_data['messages'][-1]['content']
            new_ground_truth_str = temp_data['切分子句的模型回复']
            sub_sentences = extract_clauses(new_ground_truth_str)
            assert len(ground_truth_str) == len("".join(sub_sentences))
            list_critical_ids = set(temp_data['list_critical_ids'])
            if len(list_critical_ids)==0:
                empty_reweight_count+=1
                answer_content=temp_data['r1_res']['answer_content']
                if answer_content.strip()=="":
                    answer_content = temp_data['r1_res']['reasoning_content']
                split_list1 = answer_content.split("\n")
                list_lines = []
                for line in split_list1:
                    line = line.strip()
                    if line == "":
                        continue
                    list_lines.append(line)
                list_critical_ids = []
                for i, line in enumerate(list_lines):
                    if line.startswith("<子句编号"):
                        critical_id = line.split("<子句编号")[-1].split(">")[0]
                        critical_id = int(critical_id)
                        list_critical_ids.append(critical_id)
                if len(list_critical_ids)>0:
                    correct_num+=1
            start_index = 0
            end_index = 0
            span_weights = []
            for i, sub_sentence in enumerate(sub_sentences):
                end_index += len(sub_sentence)
                if i + 1 not in list_critical_ids:
                    temp_span = {
                        "start": start_index,
                        "end": end_index,
                        "weight": 1
                    }
                else:
                    temp_span = {
                        "start": start_index,
                        "end": end_index,
                        "weight": 3
                    }
                start_index = end_index
                span_weights.append(temp_span)
            list_train_data.append(
                {
                    "dedup_key": temp_data['dedup_key'],
                    "messages": temp_data['messages'],
                    "span_weights": span_weights
                }
            )
    print(empty_reweight_count)
    print(correct_num)
    with open(os.path.join(data_dir, 'page_info_single_turn_hangqing_final_with_key_0127_token_reweight.json'), 'w') as wf:
        json.dump(list_train_data, wf,ensure_ascii=False, indent=2)

