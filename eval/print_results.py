from pathlib import Path


if __name__ == '__main__':
    for path in Path('/weka/home-griffin/weights').rglob('*results*.txt'):
        print(path.resolve())
        print(path.name)
        with open(path.resolve(), 'r') as fd:
            print(fd.read())
        print('\n' + '*' * 50 + '\n')
