from datasets import load_dataset, DatasetDict


if __name__ == '__main__':
    dataset = load_dataset('medmcqa')

    # Validation set -> Test
    # Train -> Train, Val

    train_val = dataset['train']
    train_val = train_val.shuffle(seed=1992)

    num_val = int(0.1 * len(train_val))

    val = train_val.select(range(0, num_val))
    train = train_val.select(range(num_val, len(train_val)))
    test = dataset['validation']

    print(f'Train={len(train)}, Validation={len(val)}, Test={len(test)}')

    out_data = DatasetDict({
        'train': train,
        'validation': val,
        'test': test,
    })

    out_data.save_to_disk('/weka/home-griffin/clinical_instructions/medmcqa/dataset_hf')
