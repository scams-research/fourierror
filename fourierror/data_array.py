"""A ``scipp.DataArray`` subclass that carries a :class:`CovDataArray`."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from .variable import (
    CovarianceError,
    CovVariable,
    _cov_dims,
    _matrix,
    _values_only,
    install_operand_deferral,
)

import scipp as sc

__all__ = ["CovDataArray", "covariance_data_array"]


def _plain_data(obj: Any) -> Any:
    """The C++-side data of an operand, with no covariance attached."""
    if isinstance(obj, CovDataArray):
        return obj.to_data_array()
    if isinstance(obj, CovVariable):
        return obj.to_variable()
    return obj


def _covariance_data(obj: Any) -> Any:
    """The covariance-carrying data of an operand, for the algebra."""
    if isinstance(obj, CovDataArray):
        return obj.data
    if isinstance(obj, sc.DataArray):
        return obj.data
    return obj


def _same_data(a: Any, b: Any) -> bool:
    """Whether two data arrays are backed by the same buffer.

    ``.data`` rebuilds a new ``CovVariable`` on every access, so the
    aliasing check inside :class:`CovVariable` cannot see that ``da + da``
    has the same operand twice. Compare the underlying C++ variables instead.
    """
    if a is b:
        return True
    if not isinstance(a, sc.DataArray) or not isinstance(b, sc.DataArray):
        return False
    try:
        return bool(
            np.shares_memory(
                sc.DataArray.data.fget(a).values,  # type: ignore[union-attr]
                sc.DataArray.data.fget(b).values,  # type: ignore[union-attr]
            )
        )
    except (TypeError, ValueError):
        return False


class CovDataArray(sc.DataArray):
    """A ``scipp.DataArray`` whose data carries a full covariance matrix.

    Parameters
    ----------
    data:
        The data. A :class:`CovVariable` supplies its own covariance; a
        plain ``Variable`` is taken as uncorrelated unless ``covariance`` is
        given.
    coords, masks, name:
        As for ``scipp.DataArray``.
    covariance:
        Covariance matrix, overriding one carried by ``data``.

    Examples
    --------
      >>> import numpy as np
      >>> import scipp as sc
      >>> from fourierror.variable import covariance_array
      >>> cv = covariance_array(
      ...     dims=['x'], values=[1.0, 2.0], covariance=[[0.04, 0.02], [0.02, 0.09]]
      ... )
      >>> da = CovDataArray(data=cv, coords={'x': sc.arange('x', 2.0)})
      >>> type(da.data).__name__
      'CovVariable'
      >>> round(float(da.sum('x').data.variance), 12)
      0.17

    Notes
    -----
    ``.data`` rebuilds the ``CovVariable`` on each access, which copies
    the ``N x N`` covariance. Bind it to a local name in hot loops.
    """

    def __init__(
        self,
        data: sc.Variable,
        coords: Mapping[str, sc.Variable] | None = None,
        masks: Mapping[str, sc.Variable] | None = None,
        name: str = "",
        covariance: Any = None,
    ) -> None:
        """Create the data array; see the class docstring for the parameters."""
        if covariance is not None:
            cv = CovVariable.from_variable(_plain_data(data), covariance)
        elif isinstance(data, CovVariable):
            cv = data
        else:
            cv = CovVariable.from_variable(data)
        super().__init__(
            data=cv.to_variable(),
            coords=dict(coords or {}),
            masks=dict(masks or {}),
            name=name,
        )
        self._covariance = cv.covariance

    # -- construction --------------------------------------------------------

    @classmethod
    def _from_base(cls, base: sc.DataArray, cov: sc.Variable) -> CovDataArray:
        """Attach ``cov`` to ``base``, the plain result of an operation."""
        return cls(
            data=_values_only(base.data),
            coords=dict(base.coords),
            masks=dict(base.masks),
            name=base.name,
            covariance=cov,
        )

    @classmethod
    def from_data_array(cls, da: sc.DataArray, covariance: Any = None) -> CovDataArray:
        """Build from a plain ``DataArray``, assuming no correlation by default."""
        return cls(
            data=da.data,
            coords=dict(da.coords),
            masks=dict(da.masks),
            name=da.name,
            covariance=covariance,
        )

    def to_data_array(self) -> sc.DataArray:
        """Drop the correlations, returning a plain ``DataArray``.

        The variances of the result are the diagonal of the covariance, so
        marginal uncertainties are preserved.
        """
        return sc.DataArray(
            data=sc.DataArray.data.fget(self),  # type: ignore[union-attr]
            coords=dict(self.coords),
            masks=dict(self.masks),
            name=self.name,
        )

    # -- properties ----------------------------------------------------------

    @property
    def data(self) -> CovVariable:
        """The data, as a :class:`CovVariable`."""
        return CovVariable._wrap(
            sc.DataArray.data.fget(self),  # type: ignore[union-attr]
            self._covariance,
        )

    @data.setter
    def data(self, value: sc.Variable) -> None:
        """The data, as a :class:`~fourierror.variable.CovVariable`.

        Assigning a plain ``Variable`` takes its variances as an uncorrelated
        covariance.
        """
        cv = (
            value
            if isinstance(value, CovVariable)
            else CovVariable.from_variable(value)
        )
        sc.DataArray.data.fset(self, cv.to_variable())  # type: ignore[union-attr]
        self._covariance = cv.covariance

    @property
    def covariance(self) -> sc.Variable:
        """The full covariance matrix of the data, i.e. ``self.data.covariance``.

        A copy, for the same reason as
        :attr:`CovVariable.covariance`: writing through the accessor
        would otherwise mutate internal state and leave the C++-side variances
        stale, with no error.
        """
        return self._covariance.copy()

    @covariance.setter
    def covariance(self, cov: Any) -> None:
        """The full covariance matrix of the data, i.e. ``self.data.covariance``.

        A copy, for the same reason as
        :attr:`~fourierror.variable.CovVariable.covariance`: writing through the
        accessor would otherwise mutate internal state and leave the C++-side
        variances stale, with no error. Assigning replaces the covariance and
        updates the variances to match.
        """
        self.data = CovVariable.from_variable(
            sc.DataArray.data.fget(self),  # type: ignore[union-attr]
            cov,
        )

    @property
    def correlation(self) -> sc.Variable:
        """The correlation matrix of the data."""
        return self.data.correlation

    def __repr__(self) -> str:
        """Return the scipp representation, prefixed with the class name."""
        return f"<CovDataArray>\n{sc.DataArray.__repr__(self)}"

    def _repr_html_(self) -> str:
        """Return scipp's HTML representation, without the ``scipp.`` prefix."""
        name = type(self).__name__
        return sc.DataArray._repr_html_(self).replace(
            f"<div class='sc-obj-type'>scipp.{name} ",
            f"<div class='sc-obj-type'>{name} ",
            1,
        )

    # -- arithmetic ----------------------------------------------------------

    def _binary(self, other: Any, name: str) -> CovDataArray:
        # The base class handles coord alignment and mask merging.
        """Apply the binary operator ``name`` to ``self`` and ``other``.

        The base class handles coordinate alignment and mask merging; the
        covariance is propagated by the equivalent :class:`CovVariable` operation.
        """
        base = getattr(sc.DataArray, name)(self.to_data_array(), _plain_data(other))
        left = self.data
        # Pass the *same* object on both sides when the operands alias, so the
        # covariance algebra sees them as perfectly correlated rather than
        # independent.
        right = left if _same_data(other, self) else _covariance_data(other)
        data = getattr(left, name)(right)
        return type(self)._from_base(base, data.covariance)

    def __add__(self, other: Any) -> CovDataArray:
        """Return ``self + other``, propagating the covariance."""
        return self._binary(other, "__add__")

    def __radd__(self, other: Any) -> CovDataArray:
        """Return ``other + self``, propagating the covariance."""
        return self._binary(other, "__add__")

    def __sub__(self, other: Any) -> CovDataArray:
        """Return ``self - other``, propagating the covariance."""
        return self._binary(other, "__sub__")

    def __rsub__(self, other: Any) -> CovDataArray:
        """Return ``other - self``, propagating the covariance."""
        return (-self).__add__(other)

    def __mul__(self, other: Any) -> CovDataArray:
        """Return ``self * other``, propagating the covariance."""
        return self._binary(other, "__mul__")

    def __rmul__(self, other: Any) -> CovDataArray:
        """Return ``other * self``, propagating the covariance."""
        return self._binary(other, "__mul__")

    def __truediv__(self, other: Any) -> CovDataArray:
        """Return ``self / other``, propagating the covariance."""
        return self._binary(other, "__truediv__")

    def __rtruediv__(self, other: Any) -> CovDataArray:
        """Return ``other / self``, propagating the covariance."""
        base = _plain_data(other) / self.to_data_array()
        data = self.data.__rtruediv__(_covariance_data(other))
        return type(self)._from_base(base, data.covariance)

    def __neg__(self) -> CovDataArray:
        """Return ``-self``; the covariance is unchanged."""
        return type(self)._from_base(-self.to_data_array(), self.covariance)

    def __abs__(self) -> CovDataArray:
        """Return ``abs(self)``, propagating the covariance."""
        data = abs(self.data)
        return type(self)._from_base(abs(self.to_data_array()), data.covariance)

    def __pow__(self, other: Any) -> CovDataArray:
        """Return ``self ** other`` for an exact exponent, propagating the covariance."""
        data = self.data**other
        return type(self)._from_base(self.to_data_array() ** other, data.covariance)

    def __iadd__(self, other: Any) -> CovDataArray:  # noqa: PYI034
        """Return ``self + other`` as a new object; see :meth:`CovVariable.__iadd__`."""
        return self.__add__(other)

    def __isub__(self, other: Any) -> CovDataArray:  # noqa: PYI034
        """Return ``self - other`` as a new object; see :meth:`CovVariable.__iadd__`."""
        return self.__sub__(other)

    def __imul__(self, other: Any) -> CovDataArray:  # noqa: PYI034
        """Return ``self * other`` as a new object; see :meth:`CovVariable.__iadd__`."""
        return self.__mul__(other)

    def __itruediv__(self, other: Any) -> CovDataArray:  # noqa: PYI034
        """Return ``self / other`` as a new object; see :meth:`CovVariable.__iadd__`."""
        return self.__truediv__(other)

    # -- reductions ----------------------------------------------------------

    def sum(self, dim: Any = None) -> CovDataArray:
        """Sum over ``dim`` (all dims if ``None``), including off-diagonal terms.

        Parameters
        ----------
        dim:
            Dimension, or dimensions, to sum over.

        Returns
        -------
        :
            The sum, with the covariance of the summed elements.
        """
        return self._reduce("sum", dim)

    def mean(self, dim: Any = None) -> CovDataArray:
        """Mean over ``dim`` (all dims if ``None``), including off-diagonal terms.

        Parameters
        ----------
        dim:
            Dimension, or dimensions, to average over.

        Returns
        -------
        :
            The mean, with the covariance of the averaged elements.
        """
        return self._reduce("mean", dim)

    def _reduce(self, name: str, dim: Any) -> CovDataArray:
        """Apply the reduction ``name`` to the data and the covariance."""
        plain = self.to_data_array()
        base = getattr(plain, name)() if dim is None else getattr(plain, name)(dim)
        data = getattr(self.data, name)(dim)
        return type(self)._from_base(base, data.covariance)

    # -- shape and indexing --------------------------------------------------

    def __getitem__(self, key: Any) -> CovDataArray:
        """Slice, selecting the same elements on both covariance axes.

        The selected positions are found by slicing a probe array of flat
        indices with the *same* key, so positional, label-based and boolean
        indexing all stay consistent with the base class.
        """
        base = self.to_data_array()[key]
        n = int(np.prod(self.shape, dtype=int))
        probe = sc.DataArray(
            data=sc.array(
                dims=list(self.dims),
                values=np.arange(n, dtype=np.float64).reshape(self.shape),
            ),
            coords=dict(self.coords),
            masks=dict(self.masks),
        )
        index = np.asarray(probe[key].data.values).astype(int).ravel()
        selected = _matrix(self._covariance, n)[np.ix_(index, index)]
        cov = sc.array(
            dims=list(_cov_dims(base.dims)),
            values=selected.reshape((*base.shape, *base.shape)),
            unit=self.unit**2,
        )
        return type(self)._from_base(base, cov)

    def transpose(self, dims: Sequence[str] | None = None) -> CovDataArray:
        """Transpose to ``dims``, reordering both axes of the covariance.

        Parameters
        ----------
        dims:
            The new dimension order; reversed if ``None``.
        """
        base = self.to_data_array().transpose(None if dims is None else list(dims))
        return type(self)._from_base(base, self.data.transpose(dims).covariance)

    def to(self, **kwargs: Any) -> CovDataArray:
        """Convert unit and/or dtype, scaling the covariance accordingly.

        Parameters
        ----------
        **kwargs:
            Passed to ``scipp.DataArray.to``.
        """
        base = self.to_data_array().to(**kwargs)
        return type(self)._from_base(base, self.data.to(**kwargs).covariance)

    def copy(self, deep: bool = True) -> CovDataArray:
        """Return a copy of the data array and its covariance.

        Parameters
        ----------
        deep:
            Whether to copy the underlying buffers.
        """
        return type(self)._from_base(
            self.to_data_array().copy(deep=deep), self._covariance.copy(deep=deep)
        )

    def __copy__(self) -> CovDataArray:
        """Return a shallow copy."""
        return self.copy(deep=False)

    def __deepcopy__(self, _: Any) -> CovDataArray:
        """Return a deep copy."""
        return self.copy(deep=True)


# --------------------------------------------------------------------------
# Operations that cannot preserve a covariance
# --------------------------------------------------------------------------

#: Inherited members safe to leave alone: they return metadata, a plot, or a
#: plain view that carries no correlations by design.
_SAFE_INHERITED = frozenset(
    {
        "plot",
        "underlying_size",
        "drop_coords",
        "drop_masks",
        "assign_coords",
        "assign_masks",
        "rename",
        "rename_dims",
    }
)


def _unsupported_method(name: str) -> Any:
    """Build a stub for ``name`` that raises :class:`CovarianceError`."""

    def fail(self: Any, *_: Any, **__: Any) -> Any:
        """Not supported: raises :class:`~fourierror.variable.CovarianceError`."""
        raise CovarianceError(
            f"'{name}' does not propagate a covariance matrix. Call "
            f"'.to_data_array()' first if dropping the correlations is intended."
        )

    fail.__name__ = name
    fail._covariance_unsupported = True
    fail.__qualname__ = f"CovDataArray.{name}"
    return fail


def _install_unsupported_stubs() -> None:
    """Make every un-overridden ``DataArray`` method fail loudly.

    Same reasoning as in :mod:`fourierror.variable`: a hand-written denylist lets
    anything omitted degrade silently to a plain ``DataArray``.
    """
    for name in dir(sc.DataArray):
        if name.startswith("_") or name in _SAFE_INHERITED:
            continue
        if name in vars(CovDataArray):
            continue
        if isinstance(inspect.getattr_static(sc.DataArray, name), property):
            continue
        if not callable(getattr(sc.DataArray, name, None)):
            continue
        setattr(CovDataArray, name, _unsupported_method(name))


_install_unsupported_stubs()

for _op in ("__floordiv__", "__mod__", "__and__", "__or__", "__xor__"):
    setattr(CovDataArray, _op, _unsupported_method(_op))
del _op


@functools.wraps(CovDataArray.__init__)
def covariance_data_array(*args: Any, **kwargs: Any) -> CovDataArray:
    """Create a :class:`CovDataArray`."""
    return CovDataArray(*args, **kwargs)


# Installed on import, not behind install_dispatch(): without it
# `plain_variable * cov_data_array` silently returns a plain DataArray, and a
# fix for silently wrong numbers should not depend on an opt-in call. See
# install_operand_deferral for why Python's own dispatch does not cover this.
install_operand_deferral()
