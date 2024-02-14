import pandas as pd


UMLS_MR_CONSO = '/weka/home-griffin/clinical_pile/umls/MRCONSO.RRF'


if __name__ == '__main__':
    columns = [
        'cui',
        'language',
        'ts',
        'lui',
        'stt',
        'sui',
        'ispref',
        'aui',
        'saui',
        'scui',
        'sdui',
        'sab',
        'tty',
        'code',
        'str',
        'srl',
        'suppress',
        'cvf'
    ]


    # 13,609,918
    print(f'Loading in CUIS from {UMLS_MR_CONSO}')
    cui_df = pd.read_csv(UMLS_MR_CONSO, names=columns, delimiter='|', index_col=False)

    # 8,603,906 rows
    eng = cui_df[cui_df['language'] == 'ENG']
    eng.dropna(subset=['cui'], inplace=True)

    outputs = []
    for cui, sub_df in eng.groupby('cui'):
        name_cts = sub_df['str'].value_counts().to_dict()
        names = name_cts.keys()
        cts = [name_cts[n] for n in names]
        assert max(cts) == cts[0]

        outputs.append({
            'cui': cui,
            'names': '|'.join(names),
            'name_freqs': '|'.join(map(str, cts))
        })

    outputs = pd.DataFrame(outputs)
    out_fn = '/weka/home-griffin/clinical_pile/umls/cuis.csv'
    print(f'Saving {len(outputs)} to {out_fn}')
    outputs.to_csv(out_fn, index=False)
