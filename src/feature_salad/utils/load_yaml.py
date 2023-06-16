import yaml
import json
from typing import Dict


def load_yaml(filepath: str) -> Dict:
    with open(filepath) as f:
        yaml_obj = yaml.safe_load(f)
    return yaml_obj