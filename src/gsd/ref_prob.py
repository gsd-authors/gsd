from math import comb, prod, ceil, floor

𝚷 = prod


class MetaZ(type):
    def __getitem__(cls, val):
        """
        Closed range on integers
        :param val:
        :return:
        """
        l, h = val
        return range(l, h + 1)


class ℤ(metaclass=MetaZ):
    pass


M = 5


def vmin(ψ: float) -> float:
    return (ceil(ψ) - ψ) * (ψ - floor(ψ))


def vmax(ψ: float) -> float:
    return (ψ - 1.) * (5 - ψ)


def C(ψ):
    Vmax, Vmin = vmax(ψ), vmin(ψ)
    if Vmax != Vmin:
        return 3. / 4. * Vmax / (Vmax - Vmin)
    else:
        return 1.


def _prob_k_1(ψ: float, ρ: float) -> float:
    return (M - ψ) / (M - 1) * 𝚷(((M - ψ) * ρ / (M - 1) + i * (C(ψ) - ρ)) / (ρ + i * (C(ψ) - ρ)) for i in ℤ[1, M - 2])


def _prob_k_M(ψ: float, ρ: float) -> float:
    return (ψ - 1) / (M - 1) * 𝚷(((ψ - 1) * ρ / (M - 1) + i * (C(ψ) - ρ)) / (ρ + i * (C(ψ) - ρ)) for i in ℤ[1, M - 2])


def _prob_beta_bin_k(ψ, ρ, k):
    return comb(M - 1, k - 1) * (ψ - 1) * (M - ψ) * ρ / ((M - 1) ** 2) * 𝚷(
        ((ψ - 1) * ρ / (M - 1) + i * (C(ψ) - ρ)) for i in ℤ[1, k - 2]) * 𝚷(
        ((M - ψ) * ρ / (M - 1) + j * (C(ψ) - ρ)) for j in ℤ[1, M - k - 1]) / 𝚷(
        (ρ + i * (C(ψ) - ρ)) for i in ℤ[1, M - 2])


def _prob_beta_bin(ψ: float, ρ: float, k: int) -> float:
    if k == 1:
        return _prob_k_1(ψ, ρ)
    elif k == M:
        return _prob_k_M(ψ, ρ)
    else:
        return _prob_beta_bin_k(ψ, ρ, k)


def _prob_mix(ψ: float, ρ: float, k: int) -> float:
    min_var_part = max(1 - abs(k - ψ), 0)
    if ρ == 1:
        ret = min_var_part
    else:
        ret = (ρ - C(ψ)) / (1 - C(ψ)) * min_var_part + (1 - ρ) / (1 - C(ψ)) * comb(M - 1, k - 1) * (
                (ψ - 1) / (M - 1)) ** (k - 1) * ((M - ψ) / (M - 1)) ** (M - k)
    return ret


def _prob_ρ_0(ψ, k):
    if k == 1:
        return (M - ψ) / (M - 1)
    elif k == M:
        return (ψ - 1) / (M - 1)
    else:
        return 0


def gsd_prob(ψ: float, ρ: float, k: int) -> float:
    if ρ < C(ψ):
        if ρ == 0:
            return _prob_ρ_0(ψ, k)
        return _prob_beta_bin(ψ, ρ, k)
    else:
        return _prob_mix(ψ, ρ, k)
