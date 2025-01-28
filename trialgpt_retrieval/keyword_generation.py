__author__ = "qiao"

"""
generate the search keywords for each patient
"""

import json
import sys

from openai import OpenAI
from tqdm import tqdm

client = OpenAI()


def get_keyword_generation_messages(note):
    system = 'You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. Please first summarize the main medical problems of the patient. Then generate up to 32 key conditions for searching relevant clinical trials for this patient. The key condition list should be ranked by priority. Please output only a JSON dict formatted as Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.'

    prompt = f"Here is the patient description: \n{note}\n\nJSON output:"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]

    return messages


if __name__ == "__main__":
    # the corpus: trec_2021, trec_2022, or sigir
    corpus = sys.argv[1]

    # the model index to use
    model = sys.argv[2]

    outputs = {}

    # First, count the total number of lines
    with open(f"dataset/{corpus}/queries.jsonl", "r") as f:
        total_lines = sum(1 for _ in f)

    # Now process the file with tqdm
    with open(f"dataset/{corpus}/queries.jsonl", "r") as f:
        for line in tqdm(f, total=total_lines, desc=f"Processing {corpus} queries"):
            entry = json.loads(line)
            messages = get_keyword_generation_messages(entry["text"])

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )

            output = response.choices[0].message.content
            output = output.strip("`").strip("json")

            outputs[entry["_id"]] = json.loads(output)

    # Write the final results
    with open(f"results/retrieval_keywords_{model}_{corpus}.json", "w") as f:
        json.dump(outputs, f, indent=4)

    print(f"Results saved to results/retrieval_keywords_{model}_{corpus}.json")