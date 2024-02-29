import argparse
import nomic
from datasets import load_dataset
from nomic import atlas


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Use NOMIC Atlas for data visualization.')
    parser.add_argument('--dataset_path', default='medarc/clinical_pile_v2')

    args = parser.parse_args()


    dataset = load_dataset(args.dataset_path)

    # Build a map using the map_data method
    dataset = atlas.map_data(
        data=dataset,
        indexed_field='text',
        identifier='my-organization/my-first-map',
        description='A description of the data.',
    )

    dataset.maps[0] # to view map build status
