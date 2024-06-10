__all__ = ["unsigned_bit_range_check", "signed_bit_range_check"]


def unsigned_bit_range_check(val: int, n_bits: int) -> bool:
    """Check if the given value is within the range of unsigned integer of the given number of bits.

    Args:
        val (int): unsigned the value to check
        n_bits (int): the number of bits

    Returns:
        bool: True if the value is within the range.
    """
    if val < 0:
        raise ValueError(f"val should be non-negative. Got {val}s")
    if n_bits < 0:
        raise ValueError(f"n_bits should be non-negative. Got {n_bits}")
    if val > 2**n_bits - 1:
        raise ValueError(f"val should be less than 2**n_bits. Got {val} and {n_bits}")
    return True


def signed_bit_range_check(val: int, n_bits: int) -> bool:
    """Check if the given value is within the range of signed integer of the given number of bits.

    Args:
        val (int): the signed value to check
        n_bits (int): the number of bits
    Returns:
        bool: True if the value is within the range.

    """
    if val < -(2 ** (n_bits - 1)):
        raise ValueError(
            f"val should be greater than -2**(n_bits-1). Got {val} and {n_bits}"
        )
    if val > 2 ** (n_bits - 1) - 1:
        raise ValueError(
            f"val should be less than 2**(n_bits-1) - 1. Got {val} and {n_bits}"
        )
    return True
