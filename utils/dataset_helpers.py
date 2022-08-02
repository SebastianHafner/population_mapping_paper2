from pathlib import Path
from utils import geofiles


def get_units(dataset_path: str, split: str):
    metadata_file = Path(dataset_path) / 'metadata.json'
    metadata = geofiles.load_json(metadata_file)

    units = sorted(list(metadata['census'].keys()))
    if split == 'all':
        return units
    units = [u for u in units if metadata['census'][u]['split'] == split]
    return units


if __name__ == '__main__':
    pass

