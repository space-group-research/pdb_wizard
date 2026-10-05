"""Periodic Boundary Conditions (PBC) for crystallographic unit cells."""

from __future__ import annotations

import numpy as np
from numpy import array, cos, floor, pi, sin, sqrt


class PBC:
    __slots__ = (
        "a", "b", "c", "alpha", "beta", "gamma",
        "volume", "inverse_volume",
        "basis_matrix", "reciprocal_basis_matrix",
    )

    def __init__(
        self, a: float, b: float, c: float,
        alpha: float, beta: float, gamma: float,
    ) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._compute_matrices()

    def _compute_matrices(self) -> None:
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = self.alpha, self.beta, self.gamma

        b00 = a
        b01 = 0.0
        b02 = 0.0
        b10 = b * cos(pi / 180.0 * gamma)
        b11 = b * sin(pi / 180.0 * gamma)
        b12 = 0.0
        b20 = c * cos(pi / 180.0 * beta)
        b21 = ((b * c * cos(pi / 180.0 * alpha)) - (b10 * b20)) / b11
        b22 = sqrt(max(0.0, c * c - b20 * b20 - b21 * b21))

        self.basis_matrix = array([
            [b00, b01, b02],
            [b10, b11, b12],
            [b20, b21, b22],
        ])

        vol = b00 * (b11 * b22 - b12 * b21)
        vol += b01 * (b12 * b20 - b10 * b22)
        vol += b02 * (b10 * b21 - b11 * b20)
        self.volume = vol
        self.inverse_volume = 1.0 / vol if abs(vol) > 1e-30 else 0.0

        iv = self.inverse_volume
        self.reciprocal_basis_matrix = array([
            [iv * (b11 * b22 - b12 * b21), iv * (b02 * b21 - b01 * b22), iv * (b01 * b12 - b02 * b11)],
            [iv * (b12 * b20 - b10 * b22), iv * (b00 * b22 - b02 * b20), iv * (b02 * b10 - b00 * b12)],
            [iv * (b10 * b21 - b11 * b20), iv * (b01 * b20 - b00 * b21), iv * (b00 * b11 - b01 * b10)],
        ])

    def update(
        self, a: float, b: float, c: float,
        alpha: float, beta: float, gamma: float,
    ) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._compute_matrices()

    def min_image(self, dx: np.ndarray) -> float:
        img = np.matmul(dx, self.reciprocal_basis_matrix)
        img = np.round(img)
        di = np.matmul(img, self.basis_matrix)
        dx_return = dx - di
        return float(np.sqrt(np.dot(dx_return, dx_return)))

    def wrap(self, dx: np.ndarray) -> np.ndarray:
        img = np.matmul(dx, self.reciprocal_basis_matrix)
        img = np.round(img)
        di = np.matmul(img, self.basis_matrix)
        return dx - di

    def wrap_forward(self, dx: np.ndarray) -> np.ndarray:
        img = np.matmul(dx, self.reciprocal_basis_matrix)
        img = floor(img)
        di = np.matmul(img, self.basis_matrix)
        return dx - di

    def get_all_rs_min_image(
        self, system: list,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        i_idx, j_idx = np.triu_indices(len(system), k=1)
        coords = np.array([atom.x for atom in system])
        dx = coords[i_idx] - coords[j_idx]
        frac = np.matmul(dx, self.reciprocal_basis_matrix)
        wrapped = dx - np.matmul(np.round(frac), self.basis_matrix)
        rs = np.linalg.norm(wrapped, axis=1)
        return i_idx, j_idx, rs

    def corners(self) -> np.ndarray:
        b = self.basis_matrix
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ], dtype=float)
        return verts @ b

    def edges(self) -> list[tuple[int, int]]:
        return [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5),
            (2, 4), (2, 6),
            (3, 5), (3, 6),
            (4, 7), (5, 7), (6, 7),
        ]
