# Controlled experiment artifacts

- `exploratory-preparation/`: original preparation-limited exploratory batch,
  explicitly excluded from final performance claims.
- `control/`: three paired repeats each of injected worker asymmetry, queue-depth
  admission, and FIFO/WDRR noisy-neighbor control experiments.
- `metal/`: real pinned Mac runtime; first-recurrence cache locality, fully warm
  negative control, FIFO/WDRR, and context-budget guard.
- `report.html`: self-contained combined table, also served at gateway `/experiments`.

Each trial records the exact schedule, knobs, settings, all outcomes, quality flags
and request samples. The batch index records runtime/source metadata. AB/BA ordering
and three repeats are descriptive controls, not statistical significance. Simulator
measurements are synthetic. Both Metal replicas share the same physical Apple GPU.
See `docs/interview-lab.zh-CN.md` and the result-grounded interview narrative.
