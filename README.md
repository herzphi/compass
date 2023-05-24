# compass
**Com**panion **P**roper Motion **A**nalysis **S**oftware **S**ystem

## Installation

You can install the package by installing this repository:

```bash
pip install git+https://github.com/herzphi/compass.git
```

## Usage
To calculate odds ratios of all candidates use this example:
### Candidates Table
| column name | unit |
| ----------- | ----------- |
| Main_ID | host star ID accessable by Simbad  |
| date | Gregorian date |
| dRA | mas |
| dRA_err | mas |
| dDEC | mas |
| dDEC_err | mas |
| magnitudes_column_name | mag |
| final_uuid | ID used to link observations of the same candidate |

Example table of a single candidate with the identifier to1_0:
|final_uuid|Main_ID|date|dRA|dDEC|dRA_err|dDEC_err|mag0|
| - | - | - | - | - | - | - | - |
|to1_0|HIP82545|2017-05-05|-1625.0|1897.0|6|6|17|
|to1_0|HIP82545|2018-05-05|-1624.4|1891.8|6|6|17|
|to1_0|HIP82545|2019-05-05|-1631.4|1891.4|6|6|17|
|to1_0|HIP82545|2020-05-05|-1606.7|1893.1|6|6|17|


### Example
For a given set of observational data of candidates the script can be executed by the following commands:
```python
import pandas as pd
from compass import model
from compass import helperfunctions

observation = pd.read_csv('observation.csv')
survey_object = model.Survey(observation, magnitudes_column_name)
survey_object.set_fieldstar_models(
   # Color transformed column name from Gaias G-Band.
   magnitudes_column_name_CALC,
   # Column name of the corresponding magnitude in 2MASS.
   magnitudes_column_name_2MASS,
   cone_radius = 0.3, # in degree
   binsize = 200 # Number of objects in a single magnitude bin
)
# Inflating parameters to adjust the sharp dropoff of the Gaussians.

survey_object.set_evaluated_fieldstar_models(
   sigma_cc_min = 0,
   sigma_model_min = 0
)
```
Return a pandas DataFrame containing the results by determining the `threshold` of an odds ratio by which a candidate is acccepted as true companion:
```python 
survey_object.get_true_companions(threshold=0)
```

The package contains a module `preset_plots`, which contains a couple of intereseting plots like:
```python
import pandas as pd
from compass import model, preset_plots

df_test = pd.DataFrame({
    'Main_ID': ['HIP82545', 'HIP82545'],
    'final_uuid':['id0', 'id0'],
    'date': ['2018-04-24', '2021-06-04'],
    'dRA': [-1386, -1394],
    'dRA_err': [1,1],
    'dDEC': [-989, -995],
    'dDEC_err':  [1,1],
    'mag0':[16,16],
    'mag0_err':[1,1]
})

survey_object = model.Survey(df_test, "mag0")
survey_object.set_fieldstar_models("ks_m_calc", "ks_m", cone_radius=0.1, binsize=50)
survey_object.set_evaluated_fieldstar_models(sigma_cc_min=0, sigma_model_min=0)

preset_plots.p_ratio_plot(
    survey_object.fieldstar_model_HIP82545.candidates_objects[0],
    'HIP82545',
    'band'
)
```
<img src="plots_example/p_ratio.png" alt="Image" width="500">

<img src="plots_example/diagram.png" alt="Image" width="500">

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

BSD 3-Clause License

Copyright (c) 2022,  Max Planck Institute for Astronomy
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
