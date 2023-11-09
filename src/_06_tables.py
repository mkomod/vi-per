import torch

for n in range(1):
    for p in range(3):
        for dgp in range(0, 3):
            res = torch.load(f"../results/res_{dgp}_{n}_{p}.pt")
            rm = res.mean(dim=2)
            rs = res.std(dim=2)
            print(f"{rm[0, 0]:.3f} & {rm[0, 1]:.3f} & {rm[0, 2]:.3f}")
            print(f"{rm[1, 0]:.3f} & {rm[1, 1]:.3f} & {rm[1, 2]:.3f}")
            print(f"{rm[2, 0]:.3f} & {rm[2, 1]:.3f} & {rm[2, 2]:.3f}")
            print(f"{rm[3, 0]:.3f} & {rm[3, 1]:.3f} & {rm[3, 2]:.3f}")
            print()

