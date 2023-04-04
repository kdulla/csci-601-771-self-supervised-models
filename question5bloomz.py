import os
from datasets import load_dataset
from huggingface_hub import InferenceApi

API_TOKEN = os.getenv("BLOOMZ_API_KEY")
inference = InferenceApi("bigscience/bloom", token=API_TOKEN)
dataset = load_dataset("boolq")
dataset = dataset.shuffle()  # shuffle the data

samples = []
curr_i = 0
count_yes = 0
count_no = 0
# had to decrease num samples because of input size limitations to API
while len(samples) < 6:
    curr_sample = dataset['train'][curr_i]
    if curr_sample["answer"] == True:
        if count_yes >= 3:
            curr_i += 1
            continue
        count_yes += 1
    else:
        if count_no >= 3:
            curr_i += 1
            continue
        count_no += 1
    prompt = "Passage: " + curr_sample["passage"] + "\nQuestion: " + curr_sample["question"] + "\nAnswer: "+ str(curr_sample["answer"])
    samples.append(prompt)
    curr_i += 1

prompt_train="\n".join(samples)

correct = 0

params = {
    "max_new_tokens": 3,
    "top_k": None,
    "top_p": 0.9,
    "temperature": 0.7,
    "do_sample": False,
    "seed": 42,
    "early_stopping":None,
    "no_repeat_ngram_size":None,
    "num_beams":None,
    "return_full_text":False
    }

processed = 100
for i in range(100):
    curr_sample = dataset["validation"][i]
    prompt_eval = "Passage: " + curr_sample["passage"] + "\nQuestion: " + curr_sample["question"] + "\nAnswer: "
    prompt = prompt_train + prompt_eval
    
    infer = inference(prompt)
    if type(infer) != list:
        processed -= 1
        continue
    pred = inference(prompt)[0]["generated_text"]
    
    actual = curr_sample["answer"]
    if "True" in pred:
        pred_bool = True
    elif "False" in pred:
        pred_bool = False
    else:
        pred_bool = not actual
        print(pred)
    if actual == pred_bool:
        correct += 1

print(f"Got {correct} out of {processed} correct")