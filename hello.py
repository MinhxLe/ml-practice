from torchtyping import TT


def add(x: TT["n"]) -> TT[1]:
    return x
