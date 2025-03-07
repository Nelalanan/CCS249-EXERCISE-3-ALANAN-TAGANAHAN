import nltk
import wikipedia
from nltk import bigrams, trigrams
from nltk.tokenize import word_tokenize
from collections import Counter
import math
from nltk.data import load

nltk.download('punkt')

page = wikipedia.page("Cyrillic script")
text = page.content[:1000]

tokens = [word for word in word_tokenize(text.lower()) if word.isalnum()]

unigram_counts = Counter(tokens)
bigram_list = list(bigrams(tokens))
bigram_counts = Counter(bigram_list)
trigram_list = list(trigrams(tokens))
trigram_counts = Counter(trigram_list)

bigram_probs = {bigram: count / unigram_counts[bigram[0]] for bigram, count in bigram_counts.items()}
trigram_probs = {trigram: count / bigram_counts[trigram[:2]] for trigram, count in trigram_counts.items() if trigram[:2] in bigram_counts}

def compute_perplexity(test_sentence, ngram_probs, ngram_order):
    test_tokens = word_tokenize(test_sentence.lower())

    if ngram_order == 2:
        test_ngrams = list(bigrams(test_tokens))
    elif ngram_order == 3:
        test_ngrams = list(trigrams(test_tokens))
    else:
        raise ValueError("N-gram order must be 2 (bigram) or 3 (trigram).")

    N = len(test_ngrams)
    log_prob_sum = sum(math.log(ngram_probs.get(ngram, 1e-6)) for ngram in test_ngrams)

    return math.exp(-log_prob_sum / N) if N > 0 else float('inf')

test_sentence = "The Cyrillic script is only used mainly by Eurasia and not very well known."

bigram_perplexity = compute_perplexity(test_sentence, bigram_probs, 2)
trigram_perplexity = compute_perplexity(test_sentence, trigram_probs, 3)

print(f"Bigram Model Perplexity: \"{test_sentence}\" -> Score: {bigram_perplexity:.4f}")
print(f"Trigram Model Perplexity: \"{test_sentence}\" -> Score: {trigram_perplexity:.4f}")
