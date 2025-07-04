import random
import json

dataset_path= '/Users/guoxing.lan/projects/datasets/post_training/Qwen3-SFT-Mixin/qwen3_32b_distill_1k.jsonl'

list_lines=[]
test_ratio=0.1
val_ratio=0.1

with open(dataset_path,'r',encoding="UTF-8") as rf:
    for line in rf:
        list_lines.append(line)

random.seed(123)
random.shuffle(list_lines)

test_size=max(int(len(list_lines)*test_ratio),1)
val_size=max(int(len(list_lines)*val_ratio),1)

test_lines=list_lines[0:test_size]
val_lines=list_lines[test_size:test_size+val_size]
train_lines=list_lines[test_size+val_size:]

with open(dataset_path.replace(".jsonl","_test.jsonl"),'w',encoding="UTF-8") as wf:
    for line in test_lines:
        wf.write(line)
with open(dataset_path.replace(".jsonl","_val.jsonl"),'w',encoding="UTF-8") as wf:
    for line in val_lines:
        wf.write(line)

with open(dataset_path.replace(".jsonl","_train.jsonl"),'w',encoding="UTF-8") as wf:
    for line in train_lines:
        wf.write(line)


