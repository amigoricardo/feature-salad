from pydantic import BaseModel, validator
from typing import Optional, List
from datetime import datetime


class Feature(BaseModel):
    dtype: str
    n: int = 1
    name: List = []
    made_of: str = 'words'
    between: List[int|float|str] = [0, 100]
    distinct: int = 10

    @validator('dtype')
    def validate_dtype(cls, v):
        assert v in ['datetime', 'int', 'float', 'category', 'string', 'boolean'], \
            'must be one of "datetime", "int", "float", "category", "string" and "boolean".'
        return v

    @validator('n')
    def validate_n(cls, v):
        assert v > 0, 'must be a positive integer.'
        return v

    @validator('name')
    def validate_name(cls, v, values):
        assert len(v) <= values['n'], 'too many names for "n" chosen.'
        return v

    @validator('made_of')
    def validate_made_of(cls, v):
        assert v in ['words', 'integers'], 'must be either "words" or "integers".'
        return v
    
    @validator('between')
    def validate_between(cls, v, values):
        assert len(v) == 2, 'must be an array of the format [lower_bound, upper_bound].'
        assert type(v[0]) == type(v[1]), 'bounds have different types.'
        if values['dtype'] == 'datetime':
            assert isinstance(v[0], str), 'bounds must be ISO dates.'
        if type(v[0]) == str:
            lb = datetime.fromisoformat(v[0])
            ub = datetime.fromisoformat(v[1])
            assert ub > lb, 'end date must be after start date.'
        else:
            assert v[1] > v[0], 'upper bound must be larger than lower bound.'
        return v
