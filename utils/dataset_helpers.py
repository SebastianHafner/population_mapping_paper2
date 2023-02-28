from pathlib import Path
from utils import geofiles


def get_units(dataset_path: str, split: str):
    metadata_file = Path(dataset_path) / 'metadata.json'
    metadata = geofiles.load_json(metadata_file)

    return metadata['sets'][split]


if __name__ == '__main__':
    pass

