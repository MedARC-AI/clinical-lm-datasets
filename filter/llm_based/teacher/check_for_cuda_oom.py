from glob import glob


num_errors = 0
ct = 0


for fn in glob('*.out'):
    ct += 1
    log = open(fn).read()
    if 'torch.cuda.OutOfMemoryError' in log:
        num_errors += 1
        print(f'OOM: -> {fn}')
    elif 'error' in log.lower():
        num_errors += 1
        print('The word error found in logs...')
        print(fn)
    # else:
    #     print(f'Good: -> {fn}')

print(f'{num_errors} errors found in {ct} log files')
