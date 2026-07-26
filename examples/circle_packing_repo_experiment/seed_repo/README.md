# Circle Packing Seed Repo

The evolutionary run mutates only `src/packing.py`.

The required public function is:

```python
def run_packing():
    return centers, radii, sum_radii
```

`centers` must be shape `(26, 2)`, `radii` must be shape `(26,)`, all circles
must stay inside the unit square, and no two circles may overlap.
