import os
import requests
import json
import shutil
import argparse
import concurrent.futures
import tqdm
import random
import re
import threading

write_lock = threading.Lock()
from collections import defaultdict


def is_merge_i_j(part_i,part_j,delimiters):
    # if part_i==part_j:
    #     return True
    if (part_i=="" or part_i in delimiters) and (part_j=="" or part_j in delimiters):
        return True
    if (not (part_i=="" or part_i in delimiters)) and (part_j=="" or part_j in delimiters):
        return True
    return False

## checked by me
def split_sentences(text, custom_delimiters=None):
    """
    将文本分割成句子

    Args:
        text (str): 待分割的文本
        custom_delimiters (set, optional): 自定义分隔符集合

    Returns:
        list: 分割后的句子列表，每个元素包含子句和紧跟它的分隔符
    """
    # 默认分隔符集合
    default_delimiters = {
        '。', '！', '？', '；', '\n',
        '.', '!', '?', ';',
        '，', ',','…',
        '……', '...',
        '、',
        '：', ':',
    }

    # 如果提供了自定义分隔符，则合并
    if custom_delimiters:
        delimiters = default_delimiters.union(custom_delimiters)
    else:
        delimiters = default_delimiters

    # 构建正则表达式模式
    # 转义特殊字符并构建分隔符模式
    escaped_delimiters = [re.escape(delimiter) for delimiter in delimiters]
    pattern = '|'.join(escaped_delimiters)

    # 使用正则表达式分割文本，使用捕获组来保留分隔符;"".join(parts)应当与text等价
    parts = re.split(f'({pattern})', text)

    # 重建句子，确保每个子句都包含紧跟它的分隔符
    sentences = []
    i = 0
    j=0
    temp_sentence=""
    while i<len(parts) and j<len(parts):
        j+=1
        if j>=len(parts):
            temp_sentence="".join(parts[i:j])
            sentences.append(temp_sentence)
            break
        if is_merge_i_j(parts[i],parts[j],delimiters):
            continue
        else:
            temp_sentence="".join(parts[i:j])
            sentences.append(temp_sentence)
            i=j
    return sentences


def extract_think_content(text):
    # text="<think>"+text.split("<think>")[-1]
    pattern = r'<think>(.*?)</think>'
    res_list = re.findall(pattern, text, flags=re.DOTALL)
    return res_list


def extract_answer_content(text):
    pattern = r'<answer>(.*?)</answer>'
    res_list = re.findall(pattern, text, flags=re.DOTALL)
    return res_list


def extract_tool_calls(text):
    pattern = r'<tool_call>(.*?)</tool_call>'
    res_list = re.findall(pattern, text, flags=re.DOTALL)
    return res_list


def llm_chat(url, model, messages, stop_token=None, stream=True, enable_thinking=True, generate_cfg={}, tools=[]):
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "max_tokens": 32768,
        "tools": tools
    }

    payload = {**payload, **generate_cfg}

    if stop_token:
        payload["stop"] = stop_token

    payload.setdefault("chat_template_kwargs", {})
    payload["chat_template_kwargs"]["enable_thinking"] = enable_thinking

    response = requests.post(url, headers=headers, json=payload)

    # 检查请求是否成功
    if response.status_code == 200:
        if not stream:
            return response.json()  # 返回解析后的 JSON 响应
        else:
            return response
    else:
        # 如果请求失败，打印错误信息并返回 None
        print(f"Error: Received status code {response.status_code} from server.")
        print(f"Response body: {response.text}")
        return None


def get_r1_res(list_history_messages, ground_truth_str, new_ground_truth_str):
    # assert list_history_messages[-1]['role'] == 'user'
    my_messages = [
        {"role": "system",
         "content": f'''你是一个资深的大模型对话系统评测专家。我有一条大模型的训练数据。训练数据可以分为两部分：1. 历史对话记录；2. 模型回复参考答案。
模型回复已经经过人工核验，可以认为是正确的。此外，我还将模型回复切分为多个子句，每个子句的开头有形如“<子句编号k>”这样的占位符来代表它是第k个子句，我们称之为“切分为子句的模型回复”。你的任务是帮我识别出哪些子句对于整个回复来说是关键的，关键与否的核心判断标准是：它是否能直接或者间接影响整个模型回复的正确性。
是否关键的判断维度：
1. 如果模型回复中规划了工具调用（形如类似这样的内容：<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>），那么工具调用是否正确就代表了整个模型回复正确。因此，关键的子句就是那些能够直接或者间接影响工具调用是否正确的子句（工具名称和参数都正确才算工具调用正确），这些子句不仅包括工具调用内容本身，还要包括思考/分析过程中能够推理出规划该工具调用的子句。
2. 如果模型回复是最终总结出答案，那么关键的子句就是代表这段总结答案核心语义的部分，以及思考/分析过程中能够推理出这些核心语义的子句。

你再回答时，先进行一段分析（不超过500字），然后以多行形式给出你认为关键的子句的占位符，除此以外不要包含其他内容。格式如下：
<简要分析>
<子句编号k>
<子句编号i>
'''
         },
        {
            "role": "user",
            "content": f"""
下面的json字符串是历史对话记录：
{json.dumps(list_history_messages, ensure_ascii=False)}

下面的内容是模型回复参考答案：
{ground_truth_str}

下面的内容是切分为子句的模型回复：
{new_ground_truth_str}
"""
        }
    ]
    response = llm_chat(
        # url="https://sd08pg283re52uiu6srpg.apigateway-cn-shanghai.volceapi.com/mlp/s-20250430112323-z9cfp/v1/chat/completions",
        # url="https://sd25hrvu9b9467c7s9g00.apigateway-cn-shanghai.volceapi.com/v1/chat/completions",
        # model="Qwen3-235B-A22B",
        url="https://scvbrmoqd4mur9c3nl28g.apigateway-cn-shanghai.volceapi.com/mlp/s-20250317111254-6v5pk/v1/chat/completions",
        model="deepseek_r1_fp8_stg_pd",
        messages=my_messages,
        stop_token=None,
        stream=False,
        generate_cfg={
            # 'top_p': 0.8,
            'max_input_tokens': 32768,
            'temperature': 1.0,
            'top_k': 1,
            'max_tokens': 32768
        }
    )
    if response:
        reasoning_content = response["choices"][0]["message"]["reasoning_content"]
        answer_content = response["choices"][0]["message"]["content"]

    else:
        reasoning_content = ""
        answer_content = ""

    return reasoning_content, answer_content


def req_r1_for_scoring(temp_sample, output_file):
    messages = temp_sample['messages']
    assert messages[-1]['role'] == 'assistant'

    ground_truth_str = messages[-1]['content']
    sub_sequences = split_sentences(ground_truth_str)
    new_ground_truth_str = ""
    for i, sub_sequence in enumerate(sub_sequences):
        new_ground_truth_str += f"<子句编号{i + 1}>" + sub_sequence

    req_r1_max_num = 5

    for req_r1_inex in range(0, req_r1_max_num):
        try:

            reasoning_content, answer_content = get_r1_res(messages[1:-1], ground_truth_str, new_ground_truth_str)
            answer_content = answer_content.replace("：", ":")
            temp_sample["r1_res"] = {
                "reasoning_content": reasoning_content,
                "answer_content": answer_content
            }
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
            temp_sample['list_critical_ids'] = list_critical_ids
            temp_sample['切分子句的模型回复'] = new_ground_truth_str
            with write_lock:
                with open(output_file, 'a', encoding="UTF-8") as af:
                    af.write(json.dumps(temp_sample, ensure_ascii=False) + "\n")
            break
        except Exception as e:
            print(f'''request r1 for scoring error: {e}''')


def parse_ori_res(ori_res_dict):
    messages = []
    for i, message in enumerate(ori_res_dict['data']):
        role = message['messageType'].lower()
        content = message['text']
        if role not in ["user", "assistant"]:
            print(f'''warning! unk role {role}''')
        messages.append({
            "role": role,
            "content": content
        })
    return messages


if __name__ == "__main__":
    a = "123,,456"
    sentences = split_sentences(a)
    print(sentences)

    data_dir = r'/Users/guoxing.lan/projects/datasets/LLM/sft/token_reweight_examples'

    # with open(os.path.join(data_dir, "页面感知_行情页面_单轮测试集_多轮格式_r1_judge格式.jsonl"), 'r', encoding="UTF-8") as rf:
    with open(os.path.join(data_dir, "agent_sft_serial_multi_turns3.json"), 'r',
              encoding="UTF-8") as rf:
        list_train_samples = json.load(rf)

    output_file = os.path.join(data_dir, "agent_sft_serial_multi_turns3_r1_critical_tokens.jsonl")
    set_keys = set()
    list_runned_samples = []
    list_samples_new = []
    key_name = 'channelId'
    if key_name not in list_train_samples[0]:
        key_name = 'dedup_key'
    try:
        with open(output_file, 'r', encoding="UTF-8") as rf:
            for line in rf:
                try:
                    temp_dict = json.loads(line.strip())

                    dedup_key = temp_dict[key_name]
                    set_keys.add(dedup_key)
                    list_runned_samples.append(temp_dict)
                except Exception as e2:
                    print(e2)
    except Exception as e1:
        print(e1)
    print(f'''总的待评估样本数量：{len(list_train_samples)}''')
    print(f'''已经评估完的样本：{len(set_keys)}''')

    for sample in list_train_samples:

        if sample[key_name] in set_keys:
            continue
        list_samples_new.append(sample)
    list_samples = list_samples_new
    print(f'''还需要评估的样本数量：{len(list_samples)}''')

    is_serial = True
    max_workers = 16

    if not is_serial:
        predictions = []
        print(f'准备并发执行，最大并发数：{max_workers}')
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for temp_sample in list_samples:
                future = executor.submit(req_r1_for_scoring, temp_sample, output_file
                                         )
                futures.append(future)

            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                predictions.append(future.result())
    else:
        print(f'准备串行执行')
        for i, temp_sample in enumerate(tqdm.tqdm(list_samples)):
            prediction = req_r1_for_scoring(temp_sample,
                                            output_file
                                            )