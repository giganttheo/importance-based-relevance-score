# importance-based-relevance-score
Official code for the paper ["Mitigating the Impact of Reference Quality on Evaluation of Summarization Systems with Reference-Free Metrics"](https://aclanthology.org/2024.emnlp-main.1078/), published at EMNLP 2024

To reproduce the results from the paper:
`python .\compute_correlations.py`

---
# Usage

First we need to install some `nltk` dependencies.

```python
import nltk

nltk.download('stopwords')
nltk.download('punkt_tab')
```

Then we can import a scorer, given a corpus of textual documents.

```python
from scorer import get_ibr_scorer

corpus = ["",] # list of textual documents

ibr_scorer = get_ibr_scorer(corpus) #loads the importance-based relevance scorer with default settings
```

Finally, we can get the score for a summary, with respect to a document of the corpus:

```python
summary = "" #textual summary to evaluate
i = 0 #index of the related document in the corpus

score = ibr_scorer(summary, i)
```
