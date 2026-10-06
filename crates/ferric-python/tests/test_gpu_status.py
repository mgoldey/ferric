import ferric


def test_gpu_status_shape_without_assuming_a_device():
    st = ferric.gpu_status()
    assert set(st) == {"compiled", "mode", "status", "reason", "device"}
    assert st["mode"] in {"off", "auto", "on"}
    if not st["compiled"]:
        assert st["status"] == "not_compiled" and st["device"] is None
    elif st["status"] == "ready":
        assert st["device"]["total_bytes"] > 0
    else:
        assert st["status"] == "unavailable" and st["reason"]
