import scipy as sp
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

ngram_independant_metrics = ["factual_consistency", "relevance", "coherence",
                             "fluency", "consistency", "rouge1", "rouge2",
                             "rougeL", "rougeLsum", "coverage", "density",
                             "compression_ratio", "chrf", "human-ref-based",
                             "human-prior", "human-ref-free", "human-norm-acu",
                             "gruen", "estime", "length", "llm"
                             ]

def print_system_level_correlations(scores_tokens, range_ngrams, corr_type="spearman", human_score="relevance"):
  rows = []
  mean_score_system = []
  for system in scores_tokens.keys():
    mean_score_system.append({metric: np.mean([score[metric] for score in scores_tokens[system]]) for metric in scores_tokens[system][0].keys()})
  # corrs = {"_".join(m.split("_")[:-1]): [] for m in mean_score_system[0].keys() if m.split(" ")[0] not in ngram_independant_metrics}
  corrs = {m: [] for m in mean_score_system[0].keys()}
  for metric in mean_score_system[0].keys():
    for metric_2 in mean_score_system[0].keys():
      if corr_type == "pearson":
        corr = sp.stats.pearsonr(
                      [score[metric] for score in mean_score_system],
                      [score[metric_2] for score in mean_score_system],
                      ).statistic
      elif corr_type == "spearman":
        corr = sp.stats.spearmanr(
                      [score[metric] for score in mean_score_system],
                      [score[metric_2] for score in mean_score_system],
                      ).statistic
      elif corr_type == "kendalltau":
        corr = sp.stats.kendalltau(
                      [score[metric] for score in mean_score_system],
                      [score[metric_2] for score in mean_score_system],
                      ).statistic
      # corrs.append(corr)
      corrs[metric].append(corr)

  # rows.append([metric] + corrs)
  for k in corrs.keys():
    rows.append([k] + corrs[k])
  print(tabulate(rows, headers=[f"metric"] + [f"{m}" for m in mean_score_system[0].keys()]))
  outputs = []
  for row in rows:
    for i, ngrams_to_keep in enumerate(range_ngrams):
      outputs.append(("_".join([row[0], f"{ngrams_to_keep}-n-grams-kept"]), float(row[i+1])))
  return outputs

def print_system_level_correlations_mixes(scores_tokens, range_ngrams, corr_type="spearman", human_score="relevance"):
  rows = []
  mean_score_system = []
  for system in scores_tokens.keys():
    tmp = {metric: np.mean([score[metric] for score in scores_tokens[system]]) for metric in scores_tokens[system][0].keys()}
    tmp.update({f"ours+{m2}": (1/5 if "llm" in m2 else (1/100 if "chrf" in m2 else (-1/100 if m2 == "estime" else 1))) * tmp[m2] + tmp["corpus-tokenizer-100_3-gram_tfidf_custom-score_normalized_length-penalty_adl/1"]  for m2 in ["gruen", "estime", "bertscore", "rouge1 (1 references)", "rouge2 (1 references)", "rougeL (1 references)", "chrf (1 references)", "llm"]})
    tmp.update({f"chrf+rouge": 1/100 * tmp["chrf (1 references)"] + tmp["rouge1 (1 references)"]})
    tmp.update({f"llm+{m2}": (1/100 if "chrf" in m2 else (-1/100 if m2 == "estime" else 1)) * tmp[m2] + tmp["llm"]/5  for m2 in ["gruen", "estime", "bertscore", "rouge1 (1 references)", "rouge2 (1 references)", "rougeL (1 references)", "chrf (1 references)"]})
    mean_score_system.append(tmp)
  # corrs = {"_".join(m.split("_")[:-1]): [] for m in mean_score_system[0].keys() if m.split(" ")[0] not in ngram_independant_metrics}
  corrs = {m: [] for m in mean_score_system[0].keys()}
  for metric in mean_score_system[0].keys():
    for metric_2 in mean_score_system[0].keys():
      if corr_type == "pearson":
        corr = sp.stats.pearsonr(
                      [score[metric] for score in mean_score_system],
                      [score[metric_2] for score in mean_score_system],
                      ).statistic
      elif corr_type == "spearman":
        corr = sp.stats.spearmanr(
                      [score[metric] for score in mean_score_system],
                      [score[metric_2] for score in mean_score_system],
                      ).statistic
      elif corr_type == "kendalltau":
        corr = sp.stats.kendalltau(
                      [score[metric] for score in mean_score_system],
                      [score[metric_2] for score in mean_score_system],
                      ).statistic
      # corrs.append(corr)
      corrs[metric].append(corr)

  # rows.append([metric] + corrs)
  for k in corrs.keys():
    rows.append([k] + corrs[k])
  print(tabulate(rows, headers=[f"metric"] + [f"{m}" for m in mean_score_system[0].keys()]))
  outputs = []
  for row in rows:
    for i, ngrams_to_keep in enumerate(range_ngrams):
      outputs.append(("_".join([row[0], f"{ngrams_to_keep}-n-grams-kept"]), float(row[i+1])))
  return outputs

def print_complementarity(scores_tokens, range_ngrams, corr_type="spearman", human_score="relevance"):
  rows = []
  # mean_score_system = []
  # for system in scores_tokens.keys():
  #   mean_score_system.append({metric: np.mean([score[metric] for score in scores_tokens[system]]) for metric in scores_tokens[system][0].keys()})
  lbls = []
  scores_ = [{} for _ in range(len(list(scores_tokens.values())[0]))]
  for system in scores_tokens.keys():
    for i, score in enumerate(scores_tokens[system]):
      for k, v in score.items():
        if k in scores_[i].keys():
          scores_[i][k].append(score[k])
        else:
          scores_[i][k] = [score[k]]

  scores_ = [{m: np.mean(score[m]) for m in score.keys()} for score in scores_]

  # corrs = {"_".join(m.split("_")[:-1]): [] for m in mean_score_system[0].keys() if m.split(" ")[0] not in ngram_independant_metrics}
  corrs = {m: [] for m in scores_[0].keys()}
  mat = np.zeros((len(scores_[0].keys()), len(scores_[0].keys())))
  for i, metric in enumerate(scores_[0].keys()):
    name = metric.split(" ")[0]
    if name == "corpus-tokenizer-100_3-gram_tfidf_custom-score_normalized_length-penalty_adl/1":
      name = "Ours"
    lbls.append(name)
    for j, metric_2 in enumerate(scores_[0].keys()):
      if corr_type == "pearson":
        corr = sp.stats.pearsonr(
                      [score[metric] for score in scores_],
                      [score[metric_2] for score in scores_],
                      ).statistic
      elif corr_type == "spearman":
        corr = sp.stats.spearmanr(
                      [score[metric] for score in scores_],
                      [score[metric_2] for score in scores_],
                      ).statistic
      elif corr_type == "kendalltau":
        corr = sp.stats.kendalltau(
                      [score[metric] for score in scores_],
                      [score[metric_2] for score in scores_],
                      ).statistic
      # corrs.append(corr)
      corrs[metric].append((1 - corr) / 2)
      mat[i, j] = (1 - corr) / 2

  plt.matshow(mat)
  plt.xticks(range(len(lbls)), labels=lbls, fontsize=14, rotation="vertical")
  plt.yticks(range(len(lbls)),labels=lbls, fontsize=14)
  plt.colorbar()
  plt.savefig("complementarity-summeval.png", bbox_inches='tight')
  plt.show()

  # rows.append([metric] + corrs)
  for k in corrs.keys():
    rows.append([k] + corrs[k])
  print(tabulate(rows, headers=[f"metric"] + [f"{m}" for m in scores_[0].keys()]))
  outputs = []
  for row in rows:
    for i, ngrams_to_keep in enumerate(range_ngrams):
      outputs.append(("_".join([row[0], f"{ngrams_to_keep}-n-grams-kept"]), float(row[i+1])))
  return outputs
