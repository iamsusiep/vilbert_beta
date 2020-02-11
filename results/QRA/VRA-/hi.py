import json
#[{"question_id": 0, "answer": [9.314593151988518e-23, 1.0, 3.2909810820496605e-21, 2.3160874590138645e-20], "is_correct_rationale": false, "target": 1},

with open('val_result.json') as f:
    data = json.loads(f.read())

count_correct_rationale =  6670
correct_r, incorrect_r = 0, 0
count_incorrect_rationale = 19864
for x in data:
    if x['is_correct_rationale']:
        values = x['answer']
        pred = values.index(max(values))
        target = x['target']

        if pred == target:
            correct_r += 1
    else:
        values = x['answer']
        pred = values.index(max(values))
        target = x['target']
        if pred == target:
            incorrect_r += 1

print("QRA R+:",correct_r/count_correct_rationale )
print("QRA R-:",incorrect_r/count_incorrect_rationale )
#print(correct_r, incorrect_r)
