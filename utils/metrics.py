import nltk
import collections

def bleu_score(sentence1, sentence2):
  """Calculates the BLEU score between 2 sentences.

  Args:
    sentence1: The first sentence.
    sentence2: The second sentence.

  Returns:
    The BLEU score.
  """

  tokens1 = nltk.word_tokenize(sentence1)
  tokens2 = nltk.word_tokenize(sentence2)

  ngrams1 = collections.Counter(ngram for ngram in nltk.ngrams(tokens1, 1, 4))
  ngrams2 = collections.Counter(ngram for ngram in nltk.ngrams(tokens2, 1, 4))

  clipped_counts = {}
  for ngram, count in ngrams1.items():
    if ngram in ngrams2:
      clipped_counts[ngram] = min(count, ngrams2[ngram])
    else:
      clipped_counts[ngram] = 0

  precision = sum(clipped_counts.values()) / len(ngrams1)
  brevity_penalty = min(1, len(sentence2) / len(sentence1))

  bleu_score = brevity_penalty * (1 + (1e-12)) * precision

  return bleu_score