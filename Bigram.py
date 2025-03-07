import nltk
import wikipedia
from nltk import bigrams
from nltk.tokenize import word_tokenize
from collections import Counter
import math
from nltk.data import load

nltk.download('punkt')

tokenizer = load('tokenizers/punkt/english.pickle')

text = "This is a test sentence. It should work now!"
print(tokenizer.tokenize(text))

page = wikipedia.page("Cyrillic script")
text = page.content[:1000]

tokens = [word for word in word_tokenize(text.lower()) if word.isalnum()]

unigram_counts = Counter(tokens)
print("Unigram Counts:")
for word, count in unigram_counts.items():
    print(f"{word}: {count}")

bigram_list = list(bigrams(tokens))
bigram_counts = Counter(bigram_list)
print("\nBigram Counts:")
for bigram, count in bigram_counts.items():
    print(f"{bigram}: {count}")

bigram_probs = {bigram: count / unigram_counts[bigram[0]] for bigram, count in bigram_counts.items()}
print("\nBigram Probabilities:")
for bigram, prob in bigram_probs.items():
    print(f"{bigram}: {prob:.4f}")

def compute_perplexity(test_sentence, bigram_probs, unigram_counts):
    test_tokens = word_tokenize(test_sentence.lower())
    test_bigrams = list(bigrams(test_tokens))
    
    N = len(test_bigrams)
    log_prob_sum = sum(math.log(bigram_probs.get(bigram, 1e-6)) for bigram in test_bigrams)
    
    return math.exp(-log_prob_sum / N)

test_sentence = "The Cyrillic script is only used mainly by Eurasia and not very well known."
perplexity_score = compute_perplexity(test_sentence, bigram_probs, unigram_counts)
print("\nBigram Model Perplexity:")
print(f"Test Sentence: \"{test_sentence}\" -> Score: {perplexity_score:.4f}")
