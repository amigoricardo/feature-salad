import random
import numpy as np
import nltk
from nltk.corpus import reuters
from typing import List


class RandomWords:
    def __init__(self):
        nltk.download('reuters', quiet=True)
        self.words = list(set(
            [w.lower() for w in reuters.words() if len(w)>4 and len(w)<10]
        ))
        random.shuffle(self.words)
    
    def get_words(self, n: int = 1) -> List[str]:
        """
        Get n random words
        Args:
            n (int): number of random words

        Returns:
            list(str): list of n random words
        """
        return [self.words.pop() for w in range(n)]
