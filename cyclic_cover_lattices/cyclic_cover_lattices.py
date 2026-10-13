#!/usr/bin/env python3
"""Intersection lattices of cyclic (p,1)-planes of K3 type, p >= 5.

Version 1.0, 2026-09-29. Self-contained computational companion.

RUN WITH SAGEMATH (no input files or additional packages):
    sage -python cyclic_cover_lattices.py
    sage -python cyclic_cover_lattices.py --verbose
    sage -python cyclic_cover_lattices.py --case p7_conic_lines

An independent, exact rational-arithmetic check, without SageMath:
    python3 cyclic_cover_lattices.py --arithmetic-only
This last mode does NOT compute genera or certify lattice isometry.

INPUT AND PROVENANCE
The geometric matrices are transcribed from the authors' working files:
  p=5: Matteo - Lattice_PkProject.py, cross-checked against
       Check casi Matteo e problema Pic.txt;
  p=7,11: Anna-Calcolo p=7,11.txt and the diagrams in Reticoli.pdf;
  p=13,17,19: Fede - Calcolo SAGE reticoli Zp-coverings.txt.
Row order is preserved. Indices are ZERO-BASED. For p=5 the redundant
16- and 21-curve matrices are retained, and explicit principal submatrices
are selected. No greedy basis selection or speculative overlattice is used.
The p=19 reference matrix has its erroneous (row 2, column 3) entry set to
zero (one-based numbering), making it the symmetric matrix U + K19.
The alternative 10-by-10 geometric matrix for p=7 is also checked.

CONVENTIONS AND MATHEMATICAL SCOPE
A_n and E8 are negative definite. U has Gram matrix [[0,1],[1,0]],
and Hp has Gram matrix [[(p-1)/2,1],[1,-2]] for p=5,13.
L17 and K19 use the negative-definite conventions of AST; their matrices
are explicitly included in the reference blocks below.
The program checks symmetry, evenness, rank, signature, determinant,
Smith invariants, and equality of genera with the reference lattices.
For each comparison it checks rank >= length(discriminant group) + 2;
the even indefinite uniqueness-in-genus theorem then implies isometry.
It does not construct an explicit unimodular change of basis.

The geometric interpretation of the input intersections and invariance of
the curve classes are proved in the accompanying paper, not by this code.
Likewise, equality with the full Neron-Severi lattice requires the paper's
rank argument (periods and effective moduli dimension for moving families;
cyclotomic divisibility for rigid cases). The printed period dimension is
computed from the candidate invariant rank, not a proof of dominance.
In particular this code alone does not prove generic Picard number one
for the singular covering surface.

REFERENCES
M. Artebani, A. Sarti, S. Taki, K3 surfaces with non-symplectic
  automorphisms of prime order, arXiv:0903.3481, Tables 2--7.
V. V. Nikulin, Integral symmetric bilinear forms and some of their
  applications, Theorem 1.14.2 (uniqueness in genus criterion).
SageMath IntegralLattice documentation:
https://doc.sagemath.org/html/en/reference/modules/sage/modules/free_quadratic_module_integer_symmetric.html

The program prints its SageMath version. A failed check raises an error
and returns a nonzero exit status. All computations use exact arithmetic.
"""

import argparse
from fractions import Fraction
from math import prod

# Explicit input matrices, preserving the source row order.
CASES = [{'id': 'p5_cubic_line',
  'p': 5,
  'branch': 'Smooth cubic and line',
  'dimension': 3,
  'rank': 6,
  'discriminant': (5, 5),
  'target': 'H5 + A4',
  'delete_indices': [],
  'gram': [[0, 1, 1, 0, 0, 1],
           [1, -2, 0, 0, 0, 0],
           [1, 0, -2, 1, 0, 0],
           [0, 0, 1, -2, 1, 0],
           [0, 0, 0, 1, -2, 1],
           [1, 0, 0, 0, 1, -2]],
  'reference': [[2, 1, 0, 0, 0, 0],
                [1, -2, 0, 0, 0, 0],
                [0, 0, -2, 1, 0, 0],
                [0, 0, 1, -2, 1, 0],
                [0, 0, 0, 1, -2, 1],
                [0, 0, 0, 0, 1, -2]]},
 {'id': 'p5_conic_lines',
  'p': 5,
  'branch': 'Conic and two lines',
  'dimension': 1,
  'rank': 14,
  'discriminant': (5, 5),
  'target': 'H5 + A4 + E8',
  'delete_indices': [0, 1],
  'gram': [[-2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
           [1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, -2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -2, 0, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -2, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -2]],
  'reference': [[2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, -2, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -2, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2]]},
 {'id': 'p5_four_lines',
  'p': 5,
  'branch': 'Four lines',
  'dimension': 0,
  'rank': 18,
  'discriminant': (5,),
  'target': 'H5 + E8 + E8',
  'delete_indices': [17, 19, 20],
  'gram': [[-2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
           [1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, -2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, -2, 1],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2]],
  'reference': [[2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, -2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -2, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2]]},
 {'id': 'p7_conic_lines',
  'p': 7,
  'branch': 'Conic and two lines',
  'dimension': 1,
  'rank': 10,
  'discriminant': (7, 7),
  'target': 'U(7) + E8',
  'delete_indices': [],
  'gram': [[-2, 1, 2, 0, 0, 0, 0, 0, 0, 0],
           [1, -2, 1, 0, 1, 0, 0, 1, 0, 0],
           [2, 1, -2, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, -2, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, -2, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, -2, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, -2, 0, 0, 0],
           [0, 1, 0, 0, 1, 0, 0, -2, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, -2, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, -2]],
  'reference': [[-2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -2, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, -2, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, -2, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -2, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -2, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, -2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
                [0, 0, 0, 0, 0, 0, 0, 0, 7, 0]],
  'alternative_gram': [[-2, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                       [0, -2, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, -2, 0, 1, 0, 0, 1, 0, 0],
                       [1, 0, 0, -2, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, -2, 1, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, -2, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0, 1, -2, 0, 0, 0],
                       [0, 0, 1, 0, 1, 0, 0, -2, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1, -2, 1],
                       [1, 0, 0, 0, 0, 0, 0, 0, 1, -2]]},
 {'id': 'p7_four_lines',
  'p': 7,
  'branch': 'Four lines',
  'dimension': 0,
  'rank': 16,
  'discriminant': (7,),
  'target': 'U + E8 + A6',
  'delete_indices': [],
  'gram': [[-2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, -2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, -2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 2, 0, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, -2, 1, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 1, -2, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, -2, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2]],
  'reference': [[-2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2]]},
 {'id': 'p11_four_lines',
  'p': 11,
  'branch': 'Four lines',
  'dimension': 0,
  'rank': 12,
  'discriminant': (11,),
  'target': 'U + A10',
  'delete_indices': [],
  'gram': [[-2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
           [0, -2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 2, -2, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, -2, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 1, -2, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, -2, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2]],
  'reference': [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2]]},
 {'id': 'p13_four_lines',
  'p': 13,
  'branch': 'Four lines',
  'dimension': 0,
  'rank': 10,
  'discriminant': (13,),
  'target': 'H13 + E8',
  'delete_indices': [],
  'gram': [[-2, 0, 1, 0, 0, 1, 0, 0, 0, 1],
           [0, -2, 1, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, -2, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, -2, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, -2, 1, 0, 0, 0, 0],
           [1, 0, 0, 0, 1, -2, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, -2, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, -2, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, -2, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 1, -2]],
  'reference': [[6, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -2, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, -2, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, -2, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -2, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -2, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, -2, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, -2]]},
 {'id': 'p17_four_lines',
  'p': 17,
  'branch': 'Four lines',
  'dimension': 0,
  'rank': 6,
  'discriminant': (17,),
  'target': 'U + L17',
  'delete_indices': [],
  'gram': [[-2, 4, 1, 0, 0, 1],
           [4, -2, 1, 0, 0, 0],
           [1, 1, -2, 2, 0, 0],
           [0, 0, 2, -2, 1, 0],
           [0, 0, 0, 1, -2, 0],
           [1, 0, 0, 0, 0, -2]],
  'reference': [[0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, -2, 1, 0, 1],
                [0, 0, 1, -2, 0, 0],
                [0, 0, 0, 0, -2, 1],
                [0, 0, 1, 0, 1, -4]]},
 {'id': 'p19_four_lines',
  'p': 19,
  'branch': 'Four lines',
  'dimension': 0,
  'rank': 4,
  'discriminant': (19,),
  'target': 'U + K19',
  'delete_indices': [],
  'gram': [[0, 2, 0, 1], [2, -2, 1, 0], [0, 1, -2, 0], [1, 0, 0, -2]],
  'reference': [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -10, 1], [0, 0, 1, -2]]}]

def require(condition, message):
    """Unlike assert, these checks remain active under python -O."""
    if not condition:
        raise ValueError(message)


def principal_submatrix(rows, deleted):
    n = len(rows)
    require(len(set(deleted)) == len(deleted), 'Repeated deletion index')
    require(all(0 <= i < n for i in deleted), 'Invalid deletion index')
    keep = [i for i in range(n) if i not in deleted]
    return [[rows[i][j] for j in keep] for i in keep]


def check_input(rows, name):
    n = len(rows)
    require(n > 0 and all(len(row) == n for row in rows), name + ': not square')
    require(all(isinstance(x, int) for row in rows for x in row), name + ': not integral')
    require(all(rows[i][j] == rows[j][i] for i in range(n) for j in range(n)),
            name + ': not symmetric')
    require(all(rows[i][i] % 2 == 0 for i in range(n)), name + ': not even')


def exact_determinant(rows):
    a = [[Fraction(x) for x in row] for row in rows]
    n, result = len(a), Fraction(1)
    for j in range(n):
        pivot = next((i for i in range(j, n) if a[i][j]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != j:
            a[pivot], a[j] = a[j], a[pivot]
            result = -result
        value = a[j][j]
        result *= value
        for i in range(j + 1, n):
            ratio = a[i][j] / value
            for k in range(j, n):
                a[i][k] -= ratio * a[j][k]
    return result


def exact_inertia(rows):
    """Positive, negative and zero indices via rational congruence operations."""
    a = [[Fraction(x) for x in row] for row in rows]
    positive = negative = 0
    while a:
        n = len(a)
        pivot = next((i for i in range(n) if a[i][i]), None)
        if pivot is None:
            pair = next(((i, j) for i in range(n) for j in range(i + 1, n)
                         if a[i][j]), None)
            if pair is None:
                return positive, negative, n
            i, j = pair
            # Simultaneously change row/column i: e_i -> e_i + e_j.
            for k in range(n):
                a[k][i] += a[k][j]
            for k in range(n):
                a[i][k] += a[j][k]
            pivot = i
        order = [pivot] + [i for i in range(n) if i != pivot]
        a = [[a[i][j] for j in order] for i in order]
        value = a[0][0]
        positive += int(value > 0)
        negative += int(value < 0)
        a = [[a[i][j] - a[i][0] * a[0][j] / value
              for j in range(1, n)] for i in range(1, n)]
    return positive, negative, 0


def rank_mod_prime(rows, p):
    a = [[x % p for x in row] for row in rows]
    rank = 0
    for j in range(len(a[0])):
        pivot = next((i for i in range(rank, len(a)) if a[i][j]), None)
        if pivot is None:
            continue
        a[pivot], a[rank] = a[rank], a[pivot]
        inverse = pow(a[rank][j], -1, p)
        a[rank] = [(x * inverse) % p for x in a[rank]]
        for i in range(rank + 1, len(a)):
            multiplier = a[i][j]
            a[i] = [(x - multiplier * y) % p for x, y in zip(a[i], a[rank])]
        rank += 1
    return rank


def arithmetic_check(case):
    """Verify all ranks, determinants, signatures and p-elementary groups."""
    label = case['id']
    full = case['gram']
    check_input(full, label + ' full configuration')
    r = case['rank']
    require(exact_inertia(full) == (1, r - 1, len(full) - r),
            label + ': incorrect full-configuration inertia')
    selected = principal_submatrix(full, case['delete_indices'])
    matrices = [('selected', selected), ('reference', case['reference'])]
    if 'alternative_gram' in case:
        matrices.append(('alternative', case['alternative_gram']))
    for kind, rows in matrices:
        name = label + ' ' + kind
        check_input(rows, name)
        require(len(rows) == r, name + ': incorrect size')
        require(exact_determinant(rows) == -prod(case['discriminant']),
                name + ': incorrect determinant')
        require(exact_inertia(rows) == (1, r - 1, 0), name + ': incorrect signature')
        # All expected groups are p-elementary. Order p^a and mod-p corank a
        # together imply exactly a Smith factors p (and all others 1).
        p, length = case['p'], len(case['discriminant'])
        require(tuple(case['discriminant']) == (p,) * length,
                name + ': this arithmetic group check requires p-elementarity')
        require(r - rank_mod_prime(rows, p) == length,
                name + ': incorrect discriminant group')
    require(r >= len(case['discriminant']) + 2,
            label + ': uniqueness-in-genus bound not satisfied')
    period_dimension = Fraction(22 - r, case['p'] - 1) - 1
    require(period_dimension == case['dimension'], label + ': period dimension mismatch')
    return matrices


def sage_check(case, matrices, verbose=False):
    """Compare actual genera, not just signatures and determinants."""
    from sage.all import ZZ, matrix
    from sage.modules.free_quadratic_module_integer_symmetric import IntegralLattice

    reference = IntegralLattice(matrix(ZZ, case['reference']))
    reference_genus = reference.genus()
    for kind, rows in matrices:
        gram = matrix(ZZ, rows)
        lattice = IntegralLattice(gram)
        require(tuple(lattice.signature_pair()) == (1, case['rank'] - 1),
                case['id'] + ': Sage signature mismatch')
        smith = gram.smith_form()[0]
        factors = tuple(abs(int(x)) for x in smith.diagonal() if abs(x) != 1)
        require(factors == tuple(case['discriminant']), case['id'] + ': Smith mismatch')
        require(lattice.genus() == reference_genus,
                case['id'] + ' ' + kind + ': different genus')
        if verbose:
            print('\n' + case['id'] + ' / ' + kind)
            print('Gram matrix:')
            print(gram)
            print(lattice.discriminant_group())
            print(lattice.genus())
    # The arithmetic check already verified evenness, indefiniteness and
    # rank >= length(A_L) + 2. Nikulin now turns genus equality into isometry.


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', choices=[c['id'] for c in CASES], action='append',
                        help='run only this case; may be repeated')
    parser.add_argument('--verbose', action='store_true',
                        help='print Gram matrices, discriminant forms and genera (Sage mode)')
    parser.add_argument('--arithmetic-only', action='store_true',
                        help='exact numerical checks only, without Sage or genus comparison')
    args = parser.parse_args()
    if not args.arithmetic_only:
        try:
            from sage.env import SAGE_VERSION
        except ImportError:
            parser.error('Run with sage -python, or use --arithmetic-only for partial validation.')
        print('SageMath ' + SAGE_VERSION)
        print('Full mode: exact arithmetic, Smith forms, genera and uniqueness criterion.')
    else:
        print('ARITHMETIC-ONLY MODE: genus/isometry checks are NOT performed.')
    selected_cases = [c for c in CASES if args.case is None or c['id'] in args.case]
    results = []
    for case in selected_cases:
        matrices = arithmetic_check(case)
        if not args.arithmetic_only:
            sage_check(case, matrices, args.verbose)
        results.append(case)
    print('\n{:<19} {:>4} {:>7} {:>10} {:>3}  {}'.format(
        'case', 'rank', 'det', 'disc group', 'd', 'reference lattice'))
    for c in results:
        group = ' x '.join(str(x) for x in c['discriminant'])
        print('{:<19} {:>4} {:>7} {:>10} {:>3}  {}'.format(
            c['id'], c['rank'], -prod(c['discriminant']), group,
            c['dimension'], c['target']))
    print('\n{} cases: all requested checks passed.'.format(len(results)))
    if args.arithmetic_only:
        print('Lattice names above are comparison targets; run Sage mode to verify genera.')
    else:
        print('All geometric lattices are isometric to their reference lattices by Nikulin.')
    print('The Neron-Severi and moduli conclusions use the geometric arguments in the paper.')


if __name__ == '__main__':
    main()
