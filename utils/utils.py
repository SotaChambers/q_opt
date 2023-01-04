import yaml
from typing import List, Dict

def load_yaml(path: str) -> Dict:
    """
    yamlファイルを読み込んで辞書を返す．

    Args:
        path (str): yamlファイルのパス
    Returns:
        Dict: yamlファイルの中身
    """
    with open(path, "r") as yml:
        config = yaml.safe_load(yml)

    return config