from collections import Counter
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-V") # load from internet
counter = Counter()

# find number of tokens
num_tokens = tokenizer.vocab_size
print(f"Total number of tokens: {num_tokens}")

# reserved tokens: 126084 - 126339, 255 tokens, except 126336, which is |mdm_mask|
# we need to add 2048 tokens, so we need to find additional 2048-255 = 1793 tokens
# NOTE: we actually need to find 2048 tokens

counter.update(range(0, 126080)) # make sure all tokens are counted

def update_counts(text):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    counter.update(ids)

# load dataset
ds = load_dataset("YuYu2004/Traffic-VQA")
all_questions = ds["train"]["question"]

print(f"Traffic dataset size: {len(all_questions)}")

# Traffic-VQA dataset (1042037 questions)
for question in tqdm(all_questions):
    update_counts(question)

# NuScenes (14830 chat messages)
with open("../data/nuscenes_waypoint_short_prompt_train.json", "r") as f:
    for sample in tqdm(json.load(f)):  
        for turn in sample["conversations"]:
            update_counts(turn["value"])


least_used = [tok_id for tok_id, _ in counter.most_common()[::-1][:256]]
assert max(least_used) <= 126080
print("IDs:", least_used[:10], "…", len(least_used))

# save as npy array
np.save("unused_token_ids.npy", np.array(least_used, dtype=np.int32))
