import random
import numpy as np
import pandas as pd
from .random_words import RandomWords


class FeatureSalad:
    def __init__(self, **kwargs):
        """
        Keyword Args:
            n_numerical (int): number of numerical features.
            n_categorical (int): number of categorical features.
            n_numerical_int (int): number of numerical features with integer values.
            n_categorical_int (int): number of categorical features with integer values.
            extra_numerical_types (list[str]): list of extra numerical types. All `float64` by default.
            extra_categorical_types (list[str]): list of extra categorical types. All `category` by default.
            n_samples (int): number of samples.
            start_date (str): lower bound of date column (%Y-%m-%d).
            end_date (str): upper bound of date column (%Y-%m-%d).
            date_column (str): name of the date column. Default: 'date'.
        """
        self.n_numerical = kwargs.get('n_numerical')
        self.n_categorical = kwargs.get('n_categorical')
        self.n_numerical_int = kwargs.get('n_numerical_int', 0)
        self.n_categorical_int = kwargs.get('n_categorical_int', 0)
        self.extra_numerical_types = kwargs.get('extra_numerical_types')
        self.extra_categorical_types = kwargs.get('extra_categorical_types')
        self.n_samples = kwargs.get('n_samples')
        self.start_date = kwargs.get('start_date')
        self.end_date = kwargs.get('end_date')
        self.date_column = kwargs.get('date_column', 'date')
        self.rw = RandomWords()
        self.X = pd.DataFrame()

        if self.n_categorical_int>self.n_categorical:
            raise ValueError('n_categorical_int cannot be larger than n_categorical')
        
        if self.n_numerical_int>self.n_numerical:
            raise ValueError('n_numerical_int cannot be larger than n_numerical')
    
    def generate(self) -> None:
        """Generate dataset"""
        self._add_dates()
        self._add_numerical()
        self._add_categorical()
        self._update_types()
    
    def to_parquet(self, *args, **kwargs) -> None:
        """Save dataset to parquet"""
        self.X.to_parquet(*args, **kwargs)

    def _add_dates(self) -> None:
        """Add date column to the dataset"""
        X = pd.DataFrame(
            pd.Series(
                pd.date_range(
                    start=self.start_date, 
                    end=self.end_date, 
                    periods=self.n_samples,
                    unit='s'
                )
            ),
            columns=[self.date_column]
        )
        self.X = pd.concat([self.X, X], axis=1)

        # Potentially unnecessary step to remove time from datetime visualisation.
        # Weird behaviour from Pandas:
        # %Y, %Y-%m and %Y-%m-%d will all output %Y-%m-%d dates.
        self.X[self.date_column] = pd.to_datetime(self.X[self.date_column].dt.strftime('%Y-%m-%d'))

    def _add_numerical(self):
        """Add numerical features to the dataset"""
        for _ in range(self.n_numerical+self.n_categorical_int):
            feature_values = np.transpose(
                np.random.rand(1,self.n_samples) * np.random.randint(1,100)
            )
            X = pd.DataFrame(
                feature_values,
                columns=self.rw.get_words(1),
            )
            self.X = pd.concat([self.X, X], axis=1)

    def _add_categorical(self):
        """Add categorical features to the dataset"""
        for _ in range(self.n_categorical-self.n_categorical_int):
            feature_values = self.rw.get_words(random.randint(2,30))
            sampled_feature_values = np.random.choice(feature_values, self.n_samples)
            X = pd.DataFrame(
                sampled_feature_values, 
                columns=self.rw.get_words(1), 
                dtype='category'
            )
            self.X = pd.concat([self.X, X], axis=1)
    
    def _update_types(self):
        """
        Update data types.
        """
        # Some Floats to Integers
        num_columns = list(self.X.select_dtypes(include='float64').columns)
        num_columns_to_update = random.sample(
            num_columns, 
            self.n_categorical_int+self.n_numerical_int
        )
        self.X = self.X.astype(dict.fromkeys(num_columns_to_update, 'int64'))

        # Some Integers to Categories
        int_columns = list(self.X.select_dtypes(include='int64').columns)
        int_columns_to_update = random.sample(
            int_columns, 
            self.n_categorical_int
        )
        self.X = self.X.astype(dict.fromkeys(int_columns_to_update, 'category'))
