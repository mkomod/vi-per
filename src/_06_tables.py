import torch

for dgp in range(0, 3):
    for n in range(3):
        for p in range(2, 3):
            res = torch.load(f"../results/res_{dgp}_{n}_{p}.pt")
            rm = res.mean(dim=1)
            rs = res.std(dim=2)
            print(f"{rm[0, 0]:.3f} & {rm[0, 1]:.3f} & {rm[0, 2]:.3f} & {rm[0, 3]:.3f} & {rm[0, 4]:.3f}")
            print(f"{rm[1, 0]:.3f} & {rm[1, 1]:.3f} & {rm[1, 2]:.3f} & {rm[1, 3]:.3f} & {rm[1, 4]:.3f}")
            print(f"{rm[2, 0]:.3f} & {rm[2, 1]:.3f} & {rm[2, 2]:.3f} & {rm[2, 3]:.3f} & {rm[2, 4]:.3f}")
            print(f"{rm[3, 0]:.3f} & {rm[3, 1]:.3f} & {rm[3, 2]:.3f} & {rm[3, 3]:.3f} & {rm[3, 4]:.3f}")
            print(f"{rm[4, 0]:.3f} & {rm[4, 1]:.3f} & {rm[4, 2]:.3f} & {rm[4, 3]:.3f} & {rm[4, 4]:.3f}")
            print()

