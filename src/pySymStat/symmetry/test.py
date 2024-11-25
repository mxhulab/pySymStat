from cProfile import Profile
from .group import Symmetry

if __name__ == '__main__':
    with Profile() as pr:
        for grp in ['C', 'D', 'T', 'O', 'I']: #
            for suffix in ['', '0', '1', '2', '3', '4', '7']:
                sym = grp + suffix
                print()
                print(f'Test for group {sym}.')

                try:
                    group = Symmetry(sym)
                    group.save(f'__pycache__/{sym}.txt')
                except Exception as e:
                    print(str(e))
                    continue

                group.verify()
    pr.print_stats('cumtime')
