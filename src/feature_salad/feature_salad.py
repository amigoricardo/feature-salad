import random
import numpy as np
import pandas as pd
from typing import List
from pydantic import ValidationError
import logging

from .utils.random_words import RandomWords
from .utils.feature_schema import Feature

log = logging.getLogger('feature-salad')

class FeatureSalad:
    def __init__(self, samples: int, features: List = []):
        """
        Args:
            samples (int): number of samples.
            features (list[dict]): array of feature definitions.
        """
        self.samples = samples
        self.features = []
        for feature in features:
            try:
                self.features.append(Feature(**feature))
            except ValidationError as e:
                log.error(f'In {feature}:')
                log.error(e)
        self.rw = RandomWords()
        self.X = pd.DataFrame()
    
    def generate(self) -> pd.DataFrame:
        """Generate dataset"""
        for feature in self.features:
            for n in range(feature.n):
                try:
                    name = [feature.name[n]]
                except:
                    name = self.rw.get_words(1)
                X = self._generate_column(name, feature)
                self.X = pd.concat([self.X, X], axis=1)
        return self.X
    
    def _generate_column(self, name: str, feature: Feature) -> pd.DataFrame:
        if feature.dtype == 'datetime':
            X = pd.DataFrame(
                pd.Series(
                    pd.date_range(
                        start=feature.between[0], 
                        end=feature.between[1], 
                        periods=self.samples,
                        unit='s'
                    )
                ),
                columns=name
            )

        elif feature.dtype in ['int', 'float']:
            values = np.random.uniform(*feature.between, size=(self.samples,1))
            X = pd.DataFrame(
                values, 
                columns=name
            ).astype(feature.dtype)

        elif feature.dtype in ['category', 'string']:
            if feature.made_of == 'words':
                distinct_values = self.rw.get_words(feature.distinct)
            elif feature.made_of == 'integers':
                lb = feature.between[0]
                ub = feature.between[1]+1
                distinct_values = random.sample(range(lb,ub), feature.distinct)
            values = np.random.choice(distinct_values, self.samples)
            X = pd.DataFrame(
                values, 
                columns=name, 
                dtype=feature.dtype
            )

        elif feature.dtype == 'boolean':
            values = np.random.choice([False, True], size=(self.samples, 1))
            X = pd.DataFrame(
                values,
                columns=name,
            )

        return X
