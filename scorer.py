from tqdm import tqdm
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def get_tokenizer(corpus, vocab_size=1000):
  tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
  trainer = BpeTrainer(special_tokens=["[UNK]",], vocab_size=vocab_size)
  tokenizer.pre_tokenizer = Whitespace()
  tokenizer.train_from_iterator(corpus, trainer)
  return tokenizer

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def matching_with_summary_ngrams(ngram, ngrams_in_summary):
    # is the ngram also in the summary?
    return ngram in ngrams_in_summary

class Scorer():
  def __init__(self, corpus, tokenizer, n, mode, mode_config, coverage_penalty="fixed"):

    self.corpus = corpus
    idfs = {} #inverse document frequencies
    tfs_ = [] #term frequencies
    ngrams_docs = [] #ngrams in documents
    lengths = []

    # Compute the importance scores based on the corpus
    for document in tqdm(corpus):
        tfs = {}
        #stop words filtering
        document = word_tokenize(document)
        document = ' '.join([w for w in document if not w.lower() in stop_words])

        tokenized_document = tokenizer.encode(document.lower()).ids
        lengths.append(len(tokenized_document))

        # Compute the term frequencies and document frequencies
        ngrams_doc = list(zip(*[tokenized_document[i:len(tokenized_document) + 1 + i - n] for i in range(n)]))
        ngrams_docs.append(ngrams_doc)
        for ngram in ngrams_doc:
            #add one to the term frequency
            tfs[ngram] = tfs[ngram] + 1 if ngram in tfs.keys() else 1

        #add one to the document frequencies for all the seen ngrams
        for ngram in tfs.keys():
            idfs[ngram] = idfs[ngram] + 1 if ngram in idfs.keys() else 1
        tfs_.append({**tfs})

    adl = np.mean(lengths) #average document length

    if mode=="tf-idf":
        idfs = {k: np.log(len(corpus)/idfs[k]) for k in idfs.keys()}
    elif mode=="bm-25":
        idfs = {k: np.log(1 + (len(corpus) - idfs[k] + 0.5)/(idfs[k] + 0.5)) for k in idfs.keys()}
        tfs_ = [{k: tfs[k] / (tfs[k] + mode_config["k"] * (1 + ((dl/adl) - 1) * mode_config["b"])) for k in tfs.keys()} for dl, tfs in zip(lengths, tfs_)]
    else:
        raise Exception("Mode not supported")
    
    # print(tfs_)
    # print(idfs)
    # print(ngrams_docs)
    self.values = {"n": n, "tfs_": tfs_, "idfs": idfs, "adl": adl, "tokenizer": tokenizer, "lengths": lengths, "ngram_importance": [self.compute_ngram_importance(tfs_, idfs, j, ngrams_docs, coverage_penalty=coverage_penalty) for j in range(len(corpus))]}

  def compute_ngram_importance(self, tfs_, idfs, j, ngrams_docs, coverage_penalty="fixed", overlap_penalty=.5):
    """
    Given the term frequencies and document frequency, compute the importance scores for the ngrams
    in j-th document of the corpus
    """

    # Compute the raw importance score for each ngram of the j-th document
    ngram_importance = [(tfs_[j][k] * (idfs[k]), k) for k in tfs_[j].keys()]

    # print(ngram_importance)
    # Non-Max Penalty
    # ==> rescales the importance when neighbours have higher importance score,
    # in order to focus on the more salient ngrams between overlapping ngrams (~ non-max suppression)
    # there was an issue with this: the ngrams are in the order in which they were first seen
    # but when ngrams are occurring multiple times in the same document, this step will not work as planned
    # with the legacy version. The fixed version address this
    
    revised_ngram_importance = []
    if coverage_penalty=="legacy":
      window_size = 2
      for i in range(len(ngram_importance)):
        # if it is not the most importance ngram in the window, lower its importance score
        max_importance_in_window = np.max([ngram_importance[k][0] for k in range(max(0, i - window_size), min(i + window_size, len(ngram_importance)))])
        revised_ngram_importance.append((ngram_importance[i][0] * (10) ** np.sign(ngram_importance[i][0] - max_importance_in_window), ngram_importance[i][1]))
      ngram_importance = sorted(revised_ngram_importance, key = lambda l:l[0], reverse=True)
    elif coverage_penalty=="fixed":
      window_size = 2
      ngram_to_key_id = {":".join(map(str, k)): i for i, k in enumerate(list(tfs_[j].keys()))}
      for i in range(len(ngrams_docs[j])):
        # if it is not the most importance ngram in the window, lower its importance score
        max_importance_in_window = np.max([ngram_importance[ngram_to_key_id[":".join(map(str, ngrams_docs[j][k]))]][0] for k in range(max(0, i - window_size), min(i + window_size, len(ngrams_docs[j])))])
        revised_ngram_importance.append((ngram_importance[ngram_to_key_id[":".join(map(str, ngrams_docs[j][i]))]][0] * (10) ** np.sign(ngram_importance[ngram_to_key_id[":".join(map(str, ngrams_docs[j][i]))]][0] - max_importance_in_window), ngram_importance[ngram_to_key_id[":".join(map(str, ngrams_docs[j][i]))]][1]))
      ngram_importance = sorted(revised_ngram_importance, key = lambda l:l[0], reverse=True)
    else:
        ngram_importance = sorted(ngram_importance, key = lambda l:l[0], reverse=True)

    # Vocabulary Repetition Penalty
    # ==> penalizes the importance of ngrams that contain unigrams that are frequently repeated in the document
    # ~ gives more importance to ngrams with novel vocabulary
    revised_ngram_importance = []
    unigrams_freqs = {}
    for z, ngram in ngram_importance:
        penalty = 1
        for unigram in ngram:
            if unigram in unigrams_freqs.keys():
                penalty += unigrams_freqs[unigram] * overlap_penalty
                unigrams_freqs[unigram] += 1
            else:
                unigrams_freqs[unigram] = 1
        revised_ngram_importance.append((z / penalty, ngram))

    return sorted(revised_ngram_importance, key = lambda l:l[0], reverse=True)

  def compute_score(self, tokenizer, n, summary, j, scoring_fn, normalize=True, normalize_type="ref", length_reweighting="length-penalty"):
    epsilon = 1e-6

    #stop words filtering
    summary = word_tokenize(summary)
    summary = ' '.join([w for w in summary if not w.lower() in stop_words])

    tokenized_summary = tokenizer.encode(summary.lower()).ids

    ngrams_summary = list([tuple(tokenized_summary[i:i + n]) for i in range(len(tokenized_summary) - n + 1)])
    ngrams_in_summary = set(ngrams_summary)

    score = 0
    norm = 0
    for i, (z, ngram) in enumerate(self.values["ngram_importance"][j]):
        r = i+1 #rank
        score += matching_with_summary_ngrams(ngram, ngrams_in_summary) * scoring_fn(z + epsilon, r)
        norm += scoring_fn(z + epsilon, r)

    if normalize_type == "source":
        norm = len(ngrams_in_summary)
    if length_reweighting=="length-penalty":
        # length penalty function, aimed at penalizing summaries that are >50% the length of the source
        rw = 1/(1 + np.exp((len(ngrams_summary) / (self.values["lengths"][j] + epsilon)) * 20 - 10))
    else:
        rw = 1
    
    score = rw * score / (epsilon + norm if normalize else 1)

    return score

class TokenizedText():
    def __init__(self, ids):
       self.ids = ids

class CharacterTokenizer():
    def __init__(self):
        self.__name__ = "char-tokenizer"
    def encode(self, input):
        return TokenizedText(list(input))

class SpaceTokenizer():
    def __init__(self):
        self.__name__ = "space-tokenizer"
    def encode(self, input):
        return TokenizedText(input.split(" "))

class GPT2Tokenizer():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.__name__ = "gpt2-tokenizer"
    def encode(self, input):
        return TokenizedText(self.tokenizer(input).input_ids)

def main(corpus, source_column, machine_summary_column, n=3, tokenizer_type="corpus", importance_score_type="tanh", mode="tf-idf"):
    if tokenizer_type=="corpus":
        tokenizer = get_tokenizer([doc[source_column].lower() for doc in corpus], vocab_size=100)
        tokenizer.__name__ = "corpus-tokenizer-100"
    elif tokenizer_type=="gpt2":
        tokenizer = GPT2Tokenizer()
    elif tokenizer_type=="space":
        tokenizer = SpaceTokenizer()
    elif tokenizer_type=="character":
        tokenizer = CharacterTokenizer()
    else:
        raise Exception(f"Tokenizer type {tokenizer_type} not supported")
    
    if importance_score_type == "tanh":
        scoring_fn = lambda z, r: np.tanh(z / r)
    elif importance_score_type == "constant":
        scoring_fn = lambda z,r: 1
    elif importance_score_type == "importance":
        scoring_fn = lambda z, r:z
    elif importance_score_type == "inv-rank":
        scoring_fn = lambda z, r: 1/r
    elif importance_score_type == "exp-rank":
        scoring_fn = lambda z, r:np.exp(-r)
    else:
        raise Exception(f"Importance score function {importance_score_type} not supported")

    scorer = Scorer(corpus, tokenizer, n, mode, {"k": 1.2, "b": 0.75})
    score = []
    for j, summary in enumerate(corpus[machine_summary_column]):
        # j should be the id of the source associated to the summary in the corpus
        score.append(scorer.compute_score(tokenizer, n, summary, j, scoring_fn))
    return {"ref-free-relevance-score": score}
