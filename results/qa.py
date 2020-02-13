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
# 63% should be rationale correct

for x in data:
    values = x['answer']
    if x['is_correct_rationale']:
        # if rationale is correct
        correct_rationale += 1
        if x['target'] == values.index(max(values)):
            # if answer is correct
            correct_rationale_answer +=1  
    else:
        # if rationale is wrong
        incorrect_rationale += 1
        if x['target'] == values.index(max(values)):
            # if answer is correct
            incorrect_rationale_answer += 1
    if x['target'] != values.index(max(values)):
        #if answer is wrong
        incorrect_answer += 1
        if x['is_correct_rationale']:
            # if rationale is correct
            incorrect_answer_rationale += 1
    else:
        # if the answer is correct        
        correct_answer += 1
        if x['is_correct_rationale']:
            # if rationale is correct
            correct_answer_rationale += 1
            both_correct += 1
total = correct_answer + incorrect_answer
print("Total", correct_answer + incorrect_answer)
print("Correct  answer Predicted:", correct_answer)
print("Incorrect answer Predicted",incorrect_answer)
print("Correct Rationale Predicted:", correct_rationale)
print("Incorrect Rationale Predicted",incorrect_rationale)
print("")

print("Correct rationale predicted given correct answer", correct_answer_rationale)# same
print("Correct rationale predicted given incorrect answer", incorrect_answer_rationale)
print("Correct answer predicted given correct ratioanle",correct_rationale_answer)#same
print("Correct answer predicted given incorrect ratioanle",incorrect_rationale_answer)
print("")

print ("Both answer and rationale predicted correctly", both_correct) # same
print("")
print("Percentage of answer predicted correctly (regardless of rationale)/ total:", correct_answer/total)
print("Percentage of rationale predicted correctly (regardless of answer)/ total:", correct_rationale/total)
print("Percentage of rationale predicted correctly given correct answer/ all correcly predicted answers", correct_answer_rationale/correct_answer)
print("Percentage of rationale predicted correctly given incorrect answer/ all incorrectly answers", incorrect_answer_rationale/incorrect_answer)
print("Percentage of answer predicted correctly given correct rationale / all correctly predicted rationale:", correct_rationale_answer/correct_rationale)
print("Percentage of answer Predicted correclty given incorrect rationale/ all incorrectly predicted rationale",incorrect_rationale_answer/incorrect_rationale )
print("Percetage of both rationale and answer correctly predicted/ total:", both_correct/total)
