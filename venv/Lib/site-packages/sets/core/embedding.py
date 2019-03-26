import warnings
import numpy as np
from sets.core import Step


class Embedding(Step):
    """
    Replace string words by numeric vectors using a lookup table. The default
    fallback for unknown words is the average embedding vector and a zero
    vector for falsy words.
    """

    def __init__(self, words, embeddings, depth):
        """
        Words is a list of words to embedd. Embeddings is a numpy array of same
        length. Depth is the number of dimensions to keep. All further
        dimensions are considered part of the word.
        """
        self._index = {self.key(k): i for i, k in enumerate(words)}
        if len(self._index) != len(words):
            warnings.warn('the keys of some words override each other')
        self._embeddings = np.array(embeddings)
        self._depth = depth
        self._shape = self._embeddings.shape[1:]
        self._average = self._embeddings.mean(axis=0)
        self._zeros = np.zeros(self.shape)

    @property
    def shape(self):
        return self._shape

    def __call__(self, dataset, columns=None):
        # pylint: disable=arguments-differ
        dataset = dataset.copy()
        columns = columns or dataset.columns
        for column in columns:
            dataset[column] = self._lookup_all(dataset[column])
        return dataset

    def __contains__(self, word):
        return self.key(word) in self._index

    def __getitem__(self, word):
        index = self._index[self.key(word)]
        embedding = self._embeddings[index]
        return embedding

    def key(self, word):
        # pylint: disable=no-self-use
        if isinstance(word, np.ndarray):
            return word.tostring()
        return word

    def fallback(self, word):
        if isinstance(word, np.ndarray):
            if not word.size:
                return self._zeros
        elif not word:
            return self._zeros
        return self._average

    def _lookup_all(self, array):
        array_shape = array.shape[:self._depth]
        embedded = np.empty(array_shape + self.shape)
        for index in np.ndindex(array_shape):
            embedded[index] = self._lookup(array[index])
        return embedded

    def _lookup(self, word):
        if word in self:
            return self[word]
        else:
            return self.fallback(word)
