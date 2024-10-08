from datasets import load_dataset
from tqdm import tqdm
from scorer import get_tokenizer, Scorer
import numpy as np
from correlations import print_system_level_correlations

def iterate_summeval(
    dataset,
    scorer,
    tokenizer,
    n,
    scoring_fn,
    ):
  all_scores = {}

  for j in tqdm(range(len(dataset))): #
    source = dataset[j]["text"]

    for i in range(len(dataset[j]["machine_summaries"])):
      scores = {}
      model_summary = dataset[j]["machine_summaries"][i]
      scores["relevance"] = dataset[j]["relevance"][i]

      scores_summaries = scorer.compute_score(tokenizer, n, model_summary, j, scoring_fn)

      scores["importance-based-relevance-score"] = scores_summaries
      if i in all_scores.keys():
        all_scores[i].append(scores)
      else:
        all_scores[i] = [scores]
  return all_scores


def iterate_arxivgovreport(
    dataset,
    scorer,
    tokenizer,
    n,
    scoring_fn,
    ):
  all_scores = {}

  for j in tqdm(range(len(dataset))):
    scores = {}
    # id = corpus_id_from_id[dataset[j]["dataset_id"].split("_")[-1]]
    model_summary = dataset[j]["model_summary"]
    model_type = dataset[j]["model_type"]

    scores["relevance"] = dataset[j]["relevance"]

    scores_summaries = scorer.compute_score(tokenizer, n, model_summary, j, scoring_fn)

    scores["importance-based-relevance-score"] = scores_summaries
    if model_type in all_scores.keys():
        all_scores[model_type].append(scores)
    else:
        all_scores[model_type] = [scores]
  return all_scores


for dataset_name in ["SummEval", "arXiv", "GovReport"]:

    if dataset_name=="SummEval":
        source_column = "text"
        dataset = load_dataset("mteb/summeval")["test"]
        corpus = [doc[source_column] for doc in dataset]
        iterate_fn = iterate_summeval
    elif dataset_name in ["arXiv", "GovReport"]:
        source_column = "source"
        ds_hp = load_dataset("gigant/robust_long_abstractive_human_annotation")
        ds = load_dataset("gigant/command-r-plus-generated-qa")
        dataset = [x for x in ds_hp["test"] if x["dataset"] == dataset_name]
        id_from_id = {ds["train"][j]["id"]: j for j in range(len(ds["train"]))}
        corpus = [ds["train"][id_from_id[doc["dataset_id"].split("_")[-1]]][source_column] for doc in dataset]
        iterate_fn = iterate_arxivgovreport

    tokenizer = get_tokenizer([doc.lower() for doc in corpus], vocab_size=100)
    tokenizer.__name__ = "corpus-tokenizer-100"
    mode_config={"k": 1.2, "b": 0.75}
    scoring_fn = lambda z, r: np.tanh(z / r)
    n = 3
    mode = "tf-idf"

    scorer = Scorer(corpus, tokenizer, n, mode, {"k": 1.2, "b": 0.75}, coverage_penalty="legacy")
    print("="*15 + dataset_name + "="*15)
    print_system_level_correlations(
        iterate_fn(
            dataset,
            scorer,
            tokenizer,
            n,
            scoring_fn,
            ),
        [],
        "spearman"
    )