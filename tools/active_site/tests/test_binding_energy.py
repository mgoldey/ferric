import pytest

from tools.active_site.binding_energy import check_available_memory


def test_check_available_memory_passes_with_low_threshold():
    check_available_memory(0.001)


def test_check_available_memory_raises_with_impossible_threshold():
    with pytest.raises(MemoryError):
        check_available_memory(1_000_000.0)
