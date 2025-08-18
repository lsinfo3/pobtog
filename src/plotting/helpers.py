
def affine_normalize(x):

    m = 1 / (x.max() - x.min() + 1)
    b = x.min() + 1e-10

    return m * x + b