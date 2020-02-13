import json

#filedir = 'QRA/VRA-/val_result.json'
filedir = 'QA/val_result.json'

with open(filedir) as f:
    data = json.loads(f.read())
#{"question_id": 26530, "answer": [0.9484530091285706, 0.05076855421066284, 0.0007772326935082674, 1.2299142326810397e-06], "is_correct_rationale": false, "target": 2},
for x in data:
    x['']
