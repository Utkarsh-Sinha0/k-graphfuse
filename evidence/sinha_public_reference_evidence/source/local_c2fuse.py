"""Finite-polynomial source-local execution with the existing truncation bound."""
from types import MappingProxyType

import numpy as np
from scipy.sparse import coo_array, csr_array, issparse

from sparse_c2fuse import evaluate_sparse


def prepare_local(plan):
    """Pay for a CSC index once; the original plan remains unchanged."""
    column_operator = plan["a"].tocsc(copy=True)
    for array in (column_operator.data, column_operator.indices, column_operator.indptr):
        array.setflags(write=False)
    return MappingProxyType({**plan, "column_operator": column_operator})


def evaluate_local(plan, x):
    """Return COO output after visiting only nodes reachable in retained degree.

    COO input/output avoid allocating a full n-by-q state or n-entry row index.
    Discovery can cover the entire graph; locality is an input-dependent benefit.
    """
    a = plan.get("column_operator")
    if a is None:
        raise ValueError("Call prepare_local(plan) once before local execution")
    shape = (a.shape[0], plan["float_coefficients"][0].shape[0])
    if not issparse(x) or x.shape != shape or np.issubdtype(x.dtype, np.complexfloating):
        raise ValueError(f"State must be real and sparse with shape {shape}")
    x = coo_array(x, dtype=np.float64, copy=True)
    x.sum_duplicates()
    x.eliminate_zeros()
    if not np.all(np.isfinite(x.data)):
        raise ValueError("State must be finite")
    nodes = set(int(v) for v in x.row)
    frontier = nodes.copy()
    for _ in range(plan["degree"]):
        upcoming = set()
        for column in frontier:
            start, end = int(a.indptr[column]), int(a.indptr[column+1])
            upcoming.update(int(v) for v in a.indices[start:end])
        upcoming.difference_update(nodes)
        if not upcoming:
            break
        nodes.update(upcoming)
        frontier = upcoming
    if not nodes:
        return coo_array(shape, dtype=np.float64)
    if len(nodes)*4 > a.shape[0]:
        state = np.zeros(shape)
        state[x.row, x.col] = x.data
        result = evaluate_sparse(plan, state)
        row, col = np.nonzero(result)
        return coo_array((result[row, col], (row, col)), shape=shape)
    original = np.array(sorted(nodes), dtype=np.int64)
    index = {int(node): i for i, node in enumerate(original)}
    rows, columns, values = [], [], []
    for column in original if plan["degree"] else ():
        for edge in range(int(a.indptr[column]), int(a.indptr[column+1])):
            row = index.get(int(a.indices[edge]))
            if row is not None:
                rows.append(row); columns.append(index[int(column)]); values.append(a.data[edge])
    small = csr_array((values, (rows, columns)), shape=(len(nodes), len(nodes)))
    state = np.zeros((len(nodes), shape[1]))
    for row, col, value in zip(x.row, x.col, x.data):
        state[index[int(row)], int(col)] = value
    result = evaluate_sparse({**plan, "a": small}, state)
    row, col = np.nonzero(result)
    return coo_array((result[row, col], (original[row], col)), shape=shape)
