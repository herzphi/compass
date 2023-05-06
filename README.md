# compass
**Com**panion **P**roper Motion **A**nalysis **S**oftware **S**ystem

## Installation

You can install the package by installing this repository:

```bash
pip install git+https://github.com/herzphi/compass.git
```

## Usage
To calculate odds ratios of all candidates use this example:
### Example
#### Candidates Table
| column name | unit |
| ----------- | ----------- |
| Main_ID | host star ID accessable by Simbad  |
| date | Gregorian date |
| mjd | Modified Julian Date |
| dRA | mas |
| dRA_err | mas |
| dDEC | mas |
| dDEC_err | mas |
| magnitudes_column_name | mag |
| magnitudes_column_name_err | mag |
| final_uuid | ID used to link observations of the same candidate |
For a given set of observational data of candidates the script can be executed by the following commands:
```python
import pandas as pd
from compass import model
from compass import helperfunctions

observation = pd.read_csv('observation.csv')
survey_object = model.Survey(observation, magnitudes_column_name)
# magnitudes_column_name_2MASS = column name of the corresponding magnitude in 2MASS.
# magnitudes_column_name_CALC = Color transformed column name from Gaias G-Band.
survey_object.set_fieldstar_models(
   magnitudes_column_name_CALC, 
   magnitudes_column_name_2MASS
)
# Inflating parameters to adjust the sharp dropoff of the Gaussians.
survey_object.set_evaluated_fieldstar_models(
   sigma_cc_min=0,
   sigma_model_min=0
)
```
Return a pandas DataFrame containing the results by determining the threshold of an odds ratio by which a candidate is acccepted as true companion:
```python 
survey_object.get_true_companions(threshold=0)
```

![Flow Diagram](diagram.png)
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
