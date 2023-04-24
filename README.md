# compass
Companion Proper Motion Analysis Software System

## Installation

You can install the package by installing this repository:

```bash
pip install git+https://github.com/herzphi/compass.git
```

## Usage
To calculate odds ratios of all candidates use this example:
### Example
```python
from compass.modelling import get_p_ratio_table
from compass.preset_plots import odds_ratio_sep_mag_plot
import pandas as pd

target_name = 'HIP82545'
cone_radius = .1 # degrees
# Raw Candidates Data
candidates = pd.read_csv('data/your_candidates_data.csv')
# Calculate the p_ratios and append to the modified Candidates Data
candidates_table = get_p_ratio_table(
    target_name, 
    cone_radius, 
    candidates, 
    sigma_cc_min=0,
    sigma_model_min=0
)
# Save the table
candidates_table.to_csv(
    'data/candidates_p_ratio_table.csv', 
    index=False
)
# Show results as a odds ratio vs. seperation plot
odds_ratio_sep_mag_plot(candidates_table, target_name)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

Not yet.
