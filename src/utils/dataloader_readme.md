### Data Tensor Structure (`data[t, n, f]`)

- **dim 0 (time):** sequential time steps  
- **dim 1 (node):** sensors  
- **dim 2 (feature):**  
  - `f=0` --> traffic feature (speed?)
  - `f=1` --> time of day (normalized [0,1])  
  - `f=2` --> day of week (normalized [0,1])  
  - `f=3+` --> metadata features (location, road type, region, lanes, direction, etc.)