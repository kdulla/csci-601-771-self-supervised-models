import os
import openai
from datasets import load_dataset

openai.api_key = os.getenv("OPENAI_API_KEY")

dataset = load_dataset("boolq")
dataset = dataset.shuffle()  # shuffle the data

samples = []
curr_i = 0
count_yes = 0
count_no = 0
while len(samples) < 8:
    curr_sample = dataset['train'][curr_i]
    if curr_sample["answer"] == True:
        if count_yes >= 4:
            curr_i += 1
            continue
        count_yes += 1
    else:
        if count_no >= 4:
            curr_i += 1
            continue
        count_no += 1
    prompt = "Passage: " + curr_sample["passage"] + "\nQuestion: " + curr_sample["question"] + "\nAnswer: "+ str(curr_sample["answer"])
    samples.append(prompt)
    curr_i += 1

prompt_train="\n".join(samples)

correct = 0
for i in range(30):
    curr_sample = dataset["validation"][i]
    prompt_eval = "Passage: " + curr_sample["passage"] + "\nQuestion: " + curr_sample["question"] + "\nAnswer: "

    prompt = prompt_train + prompt_eval
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    pred = response['choices'][0]['text']
    if "True" in pred:
        pred_bool = True
    elif "False" in pred:
        pred_bool = False
    else:
        print(pred)
    actual = curr_sample["answer"]
    if actual == pred_bool:
        correct += 1

print(f"Got {correct} out of 30 correct")