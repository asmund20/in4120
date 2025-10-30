# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator

from .analyzer import Analyzer
from .corpus import Corpus
from .dictionary import InMemoryDictionary
from .document import Document


class NaiveBayesClassifier:
    """
    Defines a multinomial naive Bayes text classifier. For a detailed primer, see
    https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html.
    """

    @dataclass
    class Result:
        """
        An individual classification result, as reported back to the client.
        """
        category: str  # The category that the classifier maps the buffer into.
        score: float   # The log-probability as assessed by the classifier.

    def __init__(self, training_set: Dict[str, Corpus], fields: Iterable[str], analyzer: Analyzer):
        # Used for breaking the text up into discrete classification features.
        self._analyzer = analyzer

        # The vocabulary we've seen during training.
        self._vocabulary = InMemoryDictionary()

        # Maps a category c to the logarithm of its prior probability, i.e., c maps to log(Pr(c)).
        self._priors: Dict[str, float] = {}

        # Maps a categoy c and a term t to the number count of the term in category.
        self._count: Dict[str, Dict[str, float]] = {}

        # Maps a category c and a term t to the logarithm of its category-conditioned posterior probability,
        # i.e., (c, t) maps to log(Pr(t | c)).
        self._posteriors: Dict[str, Dict[str, float]] = {}

        # Maps a category c to the denominator used when doing Laplace smoothing for the posterior probabilities.
        self._denominators: Dict[str, int] = {}

        # Train the classifier, i.e., estimate all probabilities.
        self._compute_priors(training_set)
        self._compute_vocabulary(training_set, fields)
        self._compute_posteriors(training_set, fields)

    def _compute_priors(self, training_set: Dict[str, Corpus]) -> None:
        """
        Estimates all prior probabilities (or, rather, log-probabilities) needed for
        the naive Bayes classifier.
        """
        counts = {category: len(corpus) for category, corpus in training_set.items()}
        NORMALIZATION_FACTOR = sum(counts.values())
        probabilities = {category: docs_in_category / NORMALIZATION_FACTOR for category, docs_in_category in counts.items()}
        self._priors = {category: math.log(prob)
                        for category, prob in probabilities.items()}

    def _get_terms_in_corpus(self, corpus: Corpus, fields: Iterable[str]) -> Iterator[str]:
        for document in iter(corpus):
            for term in self._get_terms_in_document(document, fields):
                yield term

    def _get_terms_in_document(self, document: Document, fields: Iterable[str]) -> Iterator[str]:
        for field in fields:
            for term in self._get_terms(document.get_field(field, "")):
                yield term

    def _compute_vocabulary(self, training_set: Dict[str, Corpus], fields: Iterable[str]) -> None:
        """
        Builds up the overall vocabulary as seen in the training set.
        """

        for corpus in training_set.values():
            for term in self._get_terms_in_corpus(corpus, fields):
                self._vocabulary.add_if_absent(term)

    def _compute_count(self, training_set: Dict[str, Corpus], fields: Iterable[str]) -> None:
        self._count = {category: {k: v for k, v in Counter(self._get_terms_in_corpus(corpus, fields)).most_common()}
                       for category, corpus in training_set.items()}

    def _compute_denominators(self, training_set: Dict[str, Corpus], fields: Iterable[str]) -> None:
        self._denominators = {category: sum(terms.values()) + len(self._vocabulary) for category, terms in self._count.items()}

    def _compute_posteriors(self, training_set: Dict[str, Corpus], fields: Iterable[str]) -> None:
        """
        Estimates all conditional probabilities (or, rather, log-probabilities) needed for
        the naive Bayes classifier.
        """
        self._compute_count(training_set, fields)
        self._compute_denominators(training_set, fields)
        self._posteriors = {category:
                            {term: self._smooth(self._count[category].get(term, 0), category)
                             for term, _ in iter(self._vocabulary)}
                            for category, corpus in training_set.items()}

    def _smooth(self, frequency: int, category: str) -> float:
        """
        Computes a smoothed log-probability, using Lapace add-one smoothing. Assumes that
        we've already computed the correct fraction denominator to use for the given category.
        """
        return math.log((frequency + 1) / self._denominators[category])

    def _get_terms(self, buffer) -> Iterator[str]:
        """
        Processes the given text buffer and returns the sequence of normalized
        terms as they appear. Both the documents in the training set and the buffers
        we classify need to be identically processed.
        """
        return (t for t, _ in self._analyzer.terms(buffer))

    def get_prior(self, category: str) -> float:
        """
        Given a category c, returns the category's prior log-probability log(Pr(c)).

        This is an internal detail having public visibility to facilitate testing.
        """
        return self._priors[category]

    def get_posterior(self, category: str, term: str) -> float:
        """
        Given a category c and a term t, returns the posterior log-probability log(Pr(t | c)).
        If the term has not been observed for the current category, use a smoothed estimate.

        This is an internal detail having public visibility to facilitate testing.
        """
        return self._posteriors[category].get(term, self._smooth(0, category))

    def classify(self, buffer: str) -> Iterator[Result]:
        """
        Classifies the given buffer according to the multinomial naive Bayes rule. The computed (score, category) pairs
        are emitted back to the client via the supplied callback sorted according to the scores. The reported scores
        are log-probabilities, to minimize numerical underflow issues. Logarithms are base e.
        """
        raise NotImplementedError("You need to implement this as part of the obligatory assignment.")
