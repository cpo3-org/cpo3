# Benchmark data

The `data/` directory contains 516 problems as gzip-compressed MPS files.
The 24 largest omitted files from the original 511-problem collection remain
excluded to keep the repository smaller.

| Collection | Directory | Problems | Source |
| --- | --- | ---: | --- |
| Netlib | `data/netlib` | 98 | [Netlib LP collection](https://www.netlib.org/lp/data/) |
| Netlib infeasible | `data/netlib-infeasible` | 29 | [Netlib infeasible LP collection](https://www.netlib.org/lp/infeas/) |
| Kennington | `data/kennington` | 16 | [Netlib Kennington collection](https://www.netlib.org/lp/data/kennington/) |
| Maros–Mészáros | `data/maros-meszaros` | 138 | [Convex QP collection](https://www.doc.ic.ac.uk/~im/) |
| QPLIB | `data/qplib` | 16 | [QPLIB](https://qplib.zib.de/) |
| MIPLIB | `data/miplib` | 219 | [MIPLIB 2017 benchmark set, version 2](https://miplib.zib.de/download.html) |

## Preparation

- Kennington files were expanded into standard MPS format.
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
