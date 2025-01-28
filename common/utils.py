import json


def load_corpus_details(corpus_path: str) -> dict:
    """
    Load corpus details into a dictionary for quick lookup.

    Args:
        corpus_path (str): Path to the corpus JSONL file.

    Returns:
        dict: A dictionary containing corpus details, keyed by NCT ID.
    """
    corpus_details = {}
    with open(corpus_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            nct_id = entry["_id"]
            metadata = entry.get("metadata", {})
            corpus_details[nct_id] = {
                "brief_title": metadata.get("brief_title", entry.get("title", "")),
                "phase": metadata.get("phase", ""),
                "drugs": metadata.get("drugs", ""),
                "drugs_list": metadata.get("drugs_list", []),
                "diseases": metadata.get("diseases", ""),
                "diseases_list": metadata.get("diseases_list", []),
                "enrollment": metadata.get("enrollment", ""),
                "inclusion_criteria": metadata.get("inclusion_criteria", ""),
                "exclusion_criteria": metadata.get("exclusion_criteria", ""),
                "brief_summary": metadata.get("brief_summary", entry.get("text", "")),
                "NCTID": nct_id
            }
    return corpus_details