# qoblib-tools

Standalone data loaders, routing solvers, QUBO builders and solution checkers from the [Qoro hackathon kit](https://github.com/QoroQuantum/pushquantum-hackathon-2026). The bundled traffic matrix comes from [QOBLIB](https://github.com/ZIB-AOPT/QOBLIB/tree/main/08-network).

From another uv project, install the wheel with `uv add /path/to/qoblib_tools-0.1.0-py3-none-any.whl`. For a source install, use `uv add /path/to/SharQ-Hackathon/packages/qoblib_tools`. Build a fresh wheel from this directory with `uv build --wheel`.

```python
import qoblib_tools as qt

demand = qt.load_network_demand(11)
print(demand.shape)
```

Network demand is included for all benchmark sizes (5–24). Other data downloads on first use. Set `QOBLIB_DATA_DIR` to choose the cache directory.
