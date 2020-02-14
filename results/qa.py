import json

#filedir = 'QRA/VRA-/val_result.json'
filedir = 'QA/home-/val_result.json'

with open(filedir) as f:
    data = json.loads(f.read())
#{"question_id": 26530, "answer": [0.9484530091285706, 0.05076855421066284, 0.0007772326935082674, 1.2299142326810397e-06], "is_correct_rationale": false, "target": 2},
a, b = 0, 0
for x in data:
    b += 1
    v = x['answer']
    pred = v.index(max(v))
    if x['target'] == pred:
        a += 1
print(a/b) 

'''

    with open('/home/suji/spring20/vilbert_beta/results/QA/home-/val_result.json') as f:
        data = json.loads(f.read())

    for x in data:
        values = x['answer']
        d[x['question_id']] = (values.index(max(values)), x['target'])
    return d
'''
