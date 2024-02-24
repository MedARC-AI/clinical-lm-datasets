from datasets import load_dataset, concatenate_datasets


if __name__ == '__main__':
    dataset = load_dataset('allenai/peS2o')

    print(dataset['train'])

    train = dataset['train'].add_column('pes2o_split', 'train')
    validation = dataset['validation'].add_column('pes2o_split', 'validation')

    full_datest = concatenate_datasets(dataset['train'], dataset['validation'])
