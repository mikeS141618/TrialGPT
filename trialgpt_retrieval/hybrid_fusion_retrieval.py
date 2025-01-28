__author__ = "qiao"

"""
Conduct the first stage retrieval by the hybrid retriever 
"""
#nltk.download('punkt')

# import tqdm
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from nltk.tokenize import word_tokenize
from transformers import AutoModel, AutoTokenizer

from corpus_index import get_bm25_corpus_index, get_medcpt_corpus_index, load_and_format_patient_descriptions

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import load_corpus_details


def parse_arguments():
    """
    Parse and validate command-line arguments.

    Returns:
        dict: A dictionary containing the parsed arguments.

    Raises:
        SystemExit: If the correct number of arguments is not provided.
    """
    if len(sys.argv) != 9:
        print("Usage: script.py <corpus> <q_type> <k> <bm25_wt> <medcpt_wt> <overwrite> <top_k> <batch_size>")
        sys.exit(1)

    return {
        "corpus": sys.argv[1],
        "q_type": sys.argv[2],
        "k": int(sys.argv[3]),
        "bm25_wt": int(sys.argv[4]),
        "medcpt_wt": int(sys.argv[5]),
        "overwrite": sys.argv[6].lower() == 'true',
        "top_k": int(sys.argv[7]),
        "batch_size": int(sys.argv[8]),
    }


def load_queries(corpus, q_type):
    """
    Load queries from the specified file.

    Args:
        corpus (str): The name of the corpus.
        q_type (str): The type of queries.

    Returns:
        dict: A dictionary containing the loaded queries.
    """
    return json.load(open(f"results/retrieval_keywords_{q_type}_{corpus}.json"))


def load_medcpt_model():
    """Load and return the MedCPT query encoder model and tokenizer."""
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    return model, tokenizer


def perform_bm25_search(bm25, bm25_nctids, conditions, N):
    """
    Perform BM25 search for each condition.

    Args:
        bm25: The BM25 index.
        bm25_nctids (list): List of NCT IDs corresponding to the BM25 index.
        conditions (list): List of condition strings to search for.
        N (int): Number of top results to return for each condition.

    Returns:
        list: A list of lists, where each inner list contains the top N NCT IDs for a condition.
    """
    return [
        bm25.get_top_n(word_tokenize(condition.lower()), bm25_nctids, n=N)
        for condition in conditions
    ]


def perform_medcpt_search(model, tokenizer, medcpt, medcpt_nctids, conditions, N):
    """
    Perform MedCPT search for each condition.

    Args:
        model: The MedCPT model.
        tokenizer: The MedCPT tokenizer.
        medcpt: The MedCPT index.
        medcpt_nctids (list): List of NCT IDs corresponding to the MedCPT index.
        conditions (list): List of condition strings to search for.
        N (int): Number of top results to return for each condition.

    Returns:
        list: A list of lists, where each inner list contains the top N NCT IDs for a condition.
    """
    with torch.no_grad():
        encoded = tokenizer(
            conditions,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=512,
        ).to("cuda")
        # encode the queries (use the [CLS] last hidden states as the representations)
        embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
        # search the Faiss index
        scores, inds = medcpt.search(embeds, k=N)

    return [[medcpt_nctids[ind] for ind in ind_list] for ind_list in inds]


def calculate_scores(bm25_results, medcpt_results, k, bm25_wt, medcpt_wt):
    """
    Calculate combined scores for BM25 and MedCPT results.

    Args:
        bm25_results (list): Results from BM25 search.
        medcpt_results (list): Results from MedCPT search.
        k (int): Parameter for score calculation.
        bm25_wt (float): Weight for BM25 scores.
        medcpt_wt (float): Weight for MedCPT scores.

    Returns:
        tuple: A tuple containing two dictionaries:
            - nctid2score: Maps NCT IDs to their total scores.
            - nctid2details: Maps NCT IDs to dictionaries containing separate BM25 and MedCPT scores.
    """
    nctid2score = {}
    nctid2details = {}
    for condition_idx, (bm25_top_nctids, medcpt_top_nctids) in enumerate(zip(bm25_results, medcpt_results)):
        condition_weight = 1 / (condition_idx + 1)

        if bm25_wt > 0:
            for rank, nctid in enumerate(bm25_top_nctids):
                bm25_score = (1 / (rank + k)) * condition_weight * bm25_wt
                if nctid not in nctid2score:
                    nctid2score[nctid] = 0
                    nctid2details[nctid] = {"bm25_score": 0, "medcpt_score": 0}
                nctid2score[nctid] += bm25_score
                nctid2details[nctid]["bm25_score"] += bm25_score

        if medcpt_wt > 0:
            for rank, nctid in enumerate(medcpt_top_nctids):
                medcpt_score = (1 / (rank + k)) * condition_weight * medcpt_wt
                if nctid not in nctid2score:
                    nctid2score[nctid] = 0
                    nctid2details[nctid] = {"bm25_score": 0, "medcpt_score": 0}
                nctid2score[nctid] += medcpt_score
                nctid2details[nctid]["medcpt_score"] += medcpt_score

    return nctid2score, nctid2details


def assign_relevance_labels(retrieved_trials, percentiles):
    """
    Assign relevance labels to retrieved trials based on score percentiles.

    Args:
        retrieved_trials (dict): Dictionary of retrieved trials with their scores.
        percentiles (list): List of two percentile values for thresholding.

    Returns:
        defaultdict: A nested defaultdict where the outer key is the query ID,
                     the inner key is the relevance label ('0', '1', or '2'),
                     and the value is a list of trials with that label.

    Note:
        The function assigns labels as follows:
        - '2': Trials with scores above the higher percentile threshold
        - '1': Trials with scores between the two percentile thresholds
        - '0': Trials with scores below the lower percentile threshold
    """
    labeled_trials = defaultdict(lambda: defaultdict(list))
    for qid, trials in retrieved_trials.items():
        scores = [t['total_score'] for t in trials]
        thresholds = np.percentile(scores, percentiles)
        for trial in trials:
            score = trial['total_score']
            if score >= thresholds[0]:
                labeled_trials[qid]['2'].append(trial)
            elif score >= thresholds[1]:
                labeled_trials[qid]['1'].append(trial)
            else:
                labeled_trials[qid]['0'].append(trial)
    return labeled_trials


def main(args):
    """
    Main function to perform hybrid fusion retrieval and generate retrieved trials.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    This function:
    1. Loads necessary data and models
    2. Performs hybrid fusion retrieval using BM25 and MedCPT
    3. Calculates combined scores for retrieved trials
    4. Assigns relevance labels to the retrieved trials
    5. Saves the final output as a JSON file
    """
    corpus_details = load_corpus_details(f"dataset/{args.corpus}/corpus.jsonl")
    bm25, bm25_nctids = get_bm25_corpus_index(args.corpus, args.overwrite)
    medcpt, medcpt_nctids = get_medcpt_corpus_index(args.corpus, args.overwrite, args.batch_size)
    patient_descriptions = load_and_format_patient_descriptions(args.corpus)
    id2queries = load_queries(args.corpus, args.q_type)
    model, tokenizer = load_medcpt_model()

    # Perform hybrid fusion retrieval
    retrieved_trials = {}
    for qid, patient_desc in patient_descriptions.items():
        conditions = id2queries[qid]["conditions"]
        if not conditions:
            retrieved_trials[qid] = []
            continue

        bm25_results = perform_bm25_search(bm25, bm25_nctids, conditions, args.top_k)
        medcpt_results = perform_medcpt_search(model, tokenizer, medcpt, medcpt_nctids, conditions, args.top_k)
        nctid2score, nctid2details = calculate_scores(bm25_results, medcpt_results, args.k, args.bm25_wt,
                                                      args.medcpt_wt)
        top_nctids = sorted(nctid2score.items(), key=lambda x: -x[1])[:args.top_k]
        retrieved_trials[qid] = [
            {
                "nct_id": nctid,
                **corpus_details[nctid],
                "total_score": score,
                "bm25_score": nctid2details[nctid]["bm25_score"],
                "medcpt_score": nctid2details[nctid]["medcpt_score"]
            }
            for nctid, score in top_nctids
        ]

    # Save detailed retrieval results
    # detailed_output_path = f"results/qid2nctids_detailed_{args.q_type}_{args.corpus}_k{args.k}_bm25wt{args.bm25_wt}_medcptwt{args.medcpt_wt}_topk{args.top_k}.json"
    # with open(detailed_output_path, 'w') as f:
    #     json.dump(retrieved_trials, f, indent=4)

    # Generate retrieved trials with relevance labels
    labeled_trials = assign_relevance_labels(retrieved_trials, [args.percentile1, args.percentile2])

    final_output = []
    for qid, trials_by_label in labeled_trials.items():
        patient_entry = {
            "patient_id": qid,
            "patient": patient_descriptions[qid],
            "0": trials_by_label['0'],
            "1": trials_by_label['1'],
            "2": trials_by_label['2']
        }
        final_output.append(patient_entry)

    # Save retrieved trials with relevance labels
    output_path = f"results/retrieved_trials.json"
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=4)

    print(f"Retrieved trials saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform hybrid fusion retrieval and generate retrieved trials")
    parser.add_argument("corpus", help="Corpus name")
    parser.add_argument("q_type", help="Query type")
    parser.add_argument("k", type=int, help="k value")
    parser.add_argument("bm25_wt", type=float, help="BM25 weight")
    parser.add_argument("medcpt_wt", type=float, help="MedCPT weight")
    parser.add_argument("overwrite", type=str, help="Overwrite existing index (true/false)")
    parser.add_argument("top_k", type=int, help="Top K value")
    parser.add_argument("batch_size", type=int, help="Batch size for MedCPT encoding")
    parser.add_argument("percentile1", type=float, help="First percentile for relevance labeling")
    parser.add_argument("percentile2", type=float, help="Second percentile for relevance labeling")

    args = parser.parse_args()

    # Convert overwrite string to boolean
    args.overwrite = args.overwrite.lower() == 'true'

    main(args)
