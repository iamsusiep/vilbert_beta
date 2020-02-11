import json

filedir = 'QRA/VRA-/val_result.json'
with open(filedir) as f:
    data = json.loads(f.read())
#{"question_id": 26530, "answer": [0.9484530091285706, 0.05076855421066284, 0.0007772326935082674, 1.2299142326810397e-06], "is_correct_rationale": false, "target": 2},
correct_answer = 0
correct_answer_rationale = 0

incorrect_answer= 0
incorrect_answer_rationale= 0
both_correct = 0

correct_rationale = 0
correct_rationale_answer = 0

incorrect_rationale= 0
incorrect_rationale_answer= 0

for x in data:
    values = x['answer']
    if x['is_correct_rationale']:
        correct_rationale += 1
        if x['target'] == values.index(max(values)):
            correct_rationale_answer +=1  
    else:
        incorrect_rationale += 1
        if x['target'] == values.index(max(values)):
            incorrect_rationale_answer += 1
        
    if x['target'] != values.index(max(values)):
        incorrect_answer += 1
        if x['is_correct_rationale']:
            incorrect_answer_rationale += 1
    else:
        correct_answer += 1
        if x['is_correct_rationale']:
            correct_answer_rationale += 1
            both_correct += 1
total = correct_answer + incorrect_answer
print("Correct Answer Predicted", correct_answer_rationale/correct_answer)
print("Incorrect Answer Predicted", incorrect_answer_rationale/incorrect_answer)
print("Correct Rationale Predicted:", correct_rationale_answer/correct_rationale)
print("Wrong Rationale Predicted",incorrect_rationale_answer/incorrect_rationale )
print("Both correct:", both_correct/total)
print("Total", correct_answer + incorrect_answer)
