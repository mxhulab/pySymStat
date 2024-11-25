from pySymStat import Symmetry

if __name__ == '__main__':
    for name in ['C2', 'C7', 'D2', 'D7', 'T', 'O', 'I1', 'I2', 'I3']:
        group = Symmetry(name)
        print("================================================================================")
        print(f"Group elements, multiplication table, inverse table and all real irreducible representations of {name}.")
        print("================================================================================")

        M, K = group.size, len(group.irreps)
        group.print()
