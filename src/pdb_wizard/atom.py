"""Unified Atom class compatible with both pdb_wizard and moltui renderers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from pdb_wizard.constants import DEFAULT_ELEMENT, ELEMENTS, Element


class Atom:
    __slots__ = (
        "name", "_element", "x", "charge", "alpha", "epsilon",
        "sigma", "c6", "c8", "c10", "id",
        "mpmc_flag", "mpmc_mol", "mpmc_res", "file_mass",
    )

    _element_cache: dict[str, "Element"] = {}

    def __init__(self, x: float, y: float, z: float, name: str) -> None:
        self.name = str(name).strip()
        self.x = np.array([float(x), float(y), float(z)])

        cached = Atom._element_cache.get(self.name)
        if cached is not None:
            self._element = cached
        else:
            element_key = "".join(c for c in self.name[:2] if c.isalpha())
            element_key = element_key.lower().capitalize()
            if element_key not in ELEMENTS:
                element_key = element_key[0] if element_key else ""
            self._element = ELEMENTS.get(element_key, DEFAULT_ELEMENT)
            Atom._element_cache[self.name] = self._element

        self.charge = 0.0
        self.alpha = 0.0
        self.epsilon = 0.0
        self.sigma = 0.0
        self.c6 = 0.0
        self.c8 = 0.0
        self.c10 = 0.0
        self.id = 0
        # Set only when read from an MPMC file: F/M flag, molecule ID, residue
        # name and the mass column (dummy sites like M or LP are massless).
        self.mpmc_flag: Optional[str] = None
        self.mpmc_mol: Optional[int] = None
        self.mpmc_res: Optional[str] = None
        self.file_mass: Optional[float] = None

    @property
    def element(self) -> Element:
        return self._element

    @property
    def position(self) -> np.ndarray:
        return self.x

    @property
    def bond_r(self) -> float:
        return self._element.bond_r

    @property
    def vdw(self) -> float:
        return self._element.vdw_radius

    @property
    def mass(self) -> float:
        return self._element.mass

    @property
    def atomic_number(self) -> int:
        return self._element.atomic_number

    def __str__(self) -> str:
        return f"Atom instance {self._element.symbol} {self.id}"
