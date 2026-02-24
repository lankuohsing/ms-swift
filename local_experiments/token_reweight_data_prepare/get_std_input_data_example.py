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
    data_dir = r'/Users/guoxing.lan/projects/datasets/LLM/sft/token_reweight_examples'
    list_data_judged_by_r1 = []
    span_weights = []
    list_train_data = []
    with open(os.path.join(data_dir, 'agent_sft_serial_multi_turns3_r1_critical_tokens.jsonl'), 'r') as rf:
        for line in rf:
            temp_data = json.loads(line.strip())
            ground_truth_str = temp_data['messages'][-1]['content']
            new_ground_truth_str = temp_data['切分子句的模型回复']
            sub_sentences = extract_clauses(new_ground_truth_str)
            assert len(ground_truth_str) == len("".join(sub_sentences))
            list_critical_ids = set(temp_data['list_critical_ids'])
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
    with open(os.path.join(data_dir, 'agent_sft_serial_multi_turns3_token_reweight.json'), 'w') as wf:
        json.dump(list_train_data, wf,ensure_ascii=False, indent=2)

