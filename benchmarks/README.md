# Benchmark data

The `data/` directory contains 486 problems as gzip-compressed MPS files.
The 25 largest files from the original 511-problem collection are omitted to
keep the repository smaller. Quadratic objectives use the `QUADOBJ` section.

| Collection | Directory | Problems | Source |
| --- | --- | ---: | --- |
| Netlib | `data/netlib` | 98 | [Netlib LP collection](https://www.netlib.org/lp/data/) |
| Kennington | `data/kennington` | 16 | [Netlib Kennington collection](https://www.netlib.org/lp/data/kennington/) |
| Maros–Mészáros | `data/maros-meszaros` | 137 | [Convex QP collection](https://www.doc.ic.ac.uk/~im/) |
| QPLIB | `data/qplib` | 16 | [QPLIB](https://qplib.zib.de/) |
| MIPLIB | `data/miplib` | 219 | [MIPLIB 2017 benchmark set, version 2](https://miplib.zib.de/download.html) |

## Preparation

- Netlib contains the main LP collection, excluding the separate infeasible set.
  Kennington files were expanded into standard MPS format.
- Maros–Mészáros files preserve the original QPS contents.
- QPLIB and MIPLIB problems are continuous relaxations: integrality is removed
  while variable bounds are preserved. Integer optima are not reference optima
  for these relaxations.
- QPLIB conversion preserves the supplied triangular Hessian coefficients
  without halving off-diagonal entries; starting points are omitted.
- Solver-reported objectives omit constant objective offsets.

## QPLIB attribution

QPLIB data is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Source: Fabio Furini et al., *QPLIB: A Library of Quadratic Programming Instances*,
Mathematical Programming Computation (2018),
[doi:10.1007/s12532-018-0147-4](https://doi.org/10.1007/s12532-018-0147-4).
The transformations applied to these instances are described above.
