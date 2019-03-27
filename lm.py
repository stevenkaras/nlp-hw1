#!/usr/bin/env python3

import re
import random
import os.path
import itertools
import pandas as pd
from math import log2
from unicodedata import normalize
from collections import Counter, deque, defaultdict

import typing
from typing import Iterable, Tuple, Deque, Iterator, Sequence, Dict
from typing import TypeVar

T = TypeVar('T')


def sliding_window(iterable: Iterable[T], n: int = 2) -> Iterator[Tuple[T, ...]]:
    """
    Efficient sliding window function
    """
    iterator: Iterator[T] = iter(iterable)
    window: Deque[T] = deque(maxlen=n)
    for i in range(n):
        window.append(next(iterator))

    yield tuple(window)

    for value in iterator:
        window.append(value)
        yield tuple(window)


def preprocess_line(line: str) -> str:
    """
    Perform standard normalization on input text for modelling
    """
    # remove any t.co urls, as they only confound analysis
    line = re.sub(r'\bhttps?://t\.co/\w{8,16}', '', line)

    # NFC unicode normalization is best for language detection, as combined glyphs give better information density
    # NFKC destroys some information by converting compatibility characters, so we don't use that.
    line = normalize('NFC', line)
    # leading and trailing whitespace don't carry much information, and are most likely artifacts of the data collection
    # process, so get rid of them
    line = line.strip()
    # force everything into uppercase, as this preserves the most information (e.g. German straub decomposes into 2 ss when lowercased)
    line = line.upper()
    # normalize the unicode again because some upper case letters are decomposed into multiple lowercase letters
    line = normalize('NFC', line)
    return line


class LM:
    START_MARKER: str = '<s>'
    END_MARKER: str = '</s>'
    UNSEEN_NGRAM = '<unk>'

    def __init__(self):
        self.unigrams: Dict[str, float] = {}
        self.bigrams: Dict[Tuple[str, ...], float] = {}
        self.trigrams: Dict[Tuple[str, ...], float] = {}

        # we haven't seen anything, so everything is unseen
        self.unigrams[self.UNSEEN_NGRAM] = 0.0

    def _line_to_tokens(self, line: str, boundary_tokens: int = 1) -> Iterator[str]:
        preamble = iter([self.START_MARKER] * boundary_tokens)
        epilogue = iter([self.END_MARKER] * boundary_tokens)
        return itertools.chain(preamble, iter(line), epilogue)

    def train(self, lines: Iterable[str]) -> None:
        unigrams: typing.Counter[str] = Counter()
        bigrams: typing.Counter[Tuple[str, ...]] = Counter()
        trigrams: typing.Counter[Tuple[str, ...]] = Counter()
        for line in lines:
            line = preprocess_line(line)
            if not line:
                continue

            unigrams.update(self._line_to_tokens(line, boundary_tokens=0))

            bigrams.update([
                chunk
                for chunk in sliding_window(self._line_to_tokens(line, boundary_tokens=1), 2)
            ])

            trigrams.update([
                chunk
                for chunk in sliding_window(self._line_to_tokens(line, boundary_tokens=2), 3)
            ])

        def _log_prob_ngrams(ngrams, result, smoothing=False):
            total = sum(ngrams.values())
            if smoothing:
                total += len(ngrams) + 1
            for ngram, count in ngrams.items():
                if smoothing:
                    count += 1
                prob = float(count) / total
                log_prob = log2(prob)
                result[ngram] = log_prob
            if smoothing:
                result[self.UNSEEN_NGRAM] = log2(1.0 / total)

        _log_prob_ngrams(unigrams, self.unigrams, smoothing=True)
        _log_prob_ngrams(bigrams, self.bigrams)
        _log_prob_ngrams(trigrams, self.trigrams)

    def to_stream(self, io: typing.TextIO) -> None:
        def _write_chunk(ngrams, file):
            for ngram, log_prob in ngrams.items():
                print(f'{"".join(ngram)}\t{round(log_prob, 3)}', file=file)

        _write_chunk(self.trigrams, io)
        print(file=io)
        _write_chunk(self.bigrams, io)
        print(file=io)
        _write_chunk(self.unigrams, io)

    def from_stream(self, io: Iterable[str]) -> None:
        ngrams: Deque[Dict] = deque([self.trigrams, self.bigrams, self.unigrams])
        current = ngrams.popleft()
        ngram_pattern = re.compile(r'({}|{}|{}|.)'.format(self.START_MARKER, self.END_MARKER, self.UNSEEN_NGRAM))
        for line in io:
            line = line.rstrip()
            if not line:
                current = ngrams.popleft()
                continue

            ngram, log_prob = line.rsplit('\t', maxsplit=1)

            words = tuple(re.findall(ngram_pattern, ngram))
            current[words] = float(log_prob)

    def _interpolated_prob(self, words: Tuple[str, str, str], weights: Tuple[float, float, float]) -> float:
        a, b, c = words
        unigram_logprob = self.unigrams.get(c)
        if unigram_logprob is None:
            unigram_logprob = self.unigrams[self.UNSEEN_NGRAM]
        unigram_prob = 2 ** unigram_logprob

        bigram_logprob = self.bigrams.get((b, c))
        if bigram_logprob is None:
            bigram_prob = 0.0
        else:
            bigram_prob = 2 ** bigram_logprob

        trigram_logprob = self.trigrams.get((a, b, c))
        if trigram_logprob is None:
            trigram_prob = 0.0
        else:
            trigram_prob = 2 ** trigram_logprob

        w3, w2, w1 = weights

        return w1 * unigram_prob + w2 * bigram_prob + w3 * trigram_prob

    def _line_entropy(self, line: str, weights: Tuple[float, float, float]) -> float:
        """calculate the entropy of a line, as defined on slide 38"""
        entropy = 0.0
        for ngram in sliding_window(self._line_to_tokens(line, boundary_tokens=2), 3):
            ngram = typing.cast(Tuple[str, str, str], ngram)
            ngram_prob = self._interpolated_prob(ngram, weights)
            entropy += ngram_prob * log2(ngram_prob)

        return -entropy

    def perplexity(self, lines: Iterable[str], weights: Tuple[float, float, float]) -> float:
        """return the perplexity of the averaged entropy, as seen in slide 49"""
        entropy = 0.0
        count = 0
        for line in lines:
            line = preprocess_line(line)
            if not line:
                continue

            line_entropy = self._line_entropy(line, weights)
            entropy += line_entropy
            count += 1

        return 2 ** (entropy / count)


def lm(corpus_file: str, model_file: str) -> None:
    model = LM()
    with open(corpus_file) as f:
        model.train(f)

    with open(model_file, 'w') as f:
        model.to_stream(f)


def eval(input_file: str, model_file: str, weights: Sequence[float]) -> float:
    # validate that the weights equals 1
    sum_weights = sum(weights)
    if 1.000001 < sum_weights or sum_weights < 0.999999:
        raise ValueError('Weights do not sum to 1')
    w3, w2, w1 = weights

    model = LM()
    with open(model_file) as f:
        model.from_stream(f)

    with open(input_file) as f:
        perplexity = model.perplexity(f, (w3, w2, w1))

    print(perplexity)

    return perplexity


def test_train_split(samples, test_split=0.1):
    test_size = int(len(samples) * test_split)
    random.shuffle(samples)
    return samples[:test_size], samples[test_size:]


def test_train_split_tweet_csv(csv_path, test_split=0.1):
    tweets = pd.read_csv(csv_path).tweet_text.tolist()
    # TODO: filter out newlines and other confounding issues
    test, train = test_train_split(tweets, test_split=test_split)
    csv_base, ext = os.path.splitext(csv_path)
    test_csv = ''.join([csv_base, '.test', ext])
    train_csv = ''.join([csv_base, '.train', ext])
    with open(test_csv, 'w') as f:
        f.write('\n'.join(test))
    with open(train_csv, 'w') as f:
        f.write('\n'.join(train))

    return test_csv, train_csv


def main():
    languages = ['en', 'es', 'fr', 'in', 'it', 'nl', 'pt', 'tl']
    for language in languages:
        csv_path = f'data/{language}.csv'
        test_train_split_tweet_csv(csv_path)
        lm(f'data/{language}.train.csv', f'data/{language}.lm')

    weights = [0.4, 0.3, 0.3]
    results = defaultdict(dict)
    for test_language in languages:
        for model_language in languages:
            perplexity = eval(f'data/{test_language}.test.csv', f'data/{model_language}.lm', weights)
            results[model_language][test_language] = perplexity

    print(pd.DataFrame.from_dict(results).to_latex())


if __name__ == '__main__':
    main()
