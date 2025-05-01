### Einsum:

>>> output = einsum(x, y, z, "a b c, c b d, a g k -> a b k")
=
output[a, b, k] = \sum_{c, d, g} x[a, b, c] * y[c, b, d] * z[a, g, k]


### Einops

Composition and Decomposition
rearrange(ims, "(b1 b2) h w c -> (b1 h) (b2 w) c ", b1=2)

reduce(ims, "b h w c -> h w", "min"), min, max, sum, prod