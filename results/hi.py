import json

filedir = 'QAR/home-/val_result.json'
with open(filedir) as f:
    data = json.loads(f.read())
reasoning_correct = 0
both_correct = 0
total = 0

answer_correct = 0

for x in data:
    values = x['answer']
    total += 1
    if x['is_correct_answer']:
        answer_correct += 1
    if x['target'] == values.index(max(values)): 
        reasoning_correct += 1
        if x['is_correct_answer']:
            both_correct += 1
print("Answer:", answer_correct/total)
print("Both correct:", both_correct/total)
print("reasoning correct:", reasoning_correct/total);
print("Total", total)
