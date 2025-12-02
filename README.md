# Available data:

### Training data (1982-2020)

For each crop (maize and wheat)
Five files containing climate variables from 30 days before and 210 days after planting date
  rsds - daily short-wave radiation
  pr - daily precipitation
  tas - daily mean temperature
  tmax - daily maximum temperature
  tmin - daily minimum temperature
One “solutions” file containing the target variable (yield)
Soil
One file containing soil texture, real year, nitrogen fertilization rate and CO2 concentration

### Test data (2021-2098)

For each crop (maize and wheat)
Five files containing climate variables from 30 days before and 210 days after planting date
  rsds - daily short-wave radiation
  pr - daily precipitation
  tas - daily mean temperature
  tmax - daily maximum temperature
  tmin - daily minimum temperature

#### Soil
One file containing soil texture, real year, nitrogen fertilization rate and CO2 concentration
sample_submission.csv - an example of how your submitted predictions file should look.
Important note: To make it easier for participants to work with, we have split most of the data files between the two crops (maize and wheat). However, submission files should include predictions for both crops.

#### Columns
All files contain an ID column, which is a unique ID for each datapoint (crop, gridcell and year). Apart from train_crop_solutions.csv, all other files also contain columns lat, lon, year and crop. Note that year is not the 'real year' but reflects the number of growing seasons since the start of the simulation run. However, the 'real year' can be found in the soil_co2_*.parquet files in the column real_year. In these files, texture_class contains the dominant USDA soil texture class (categorical variable coded as 1 to 13) and co2 contains the ambient CO2 concentration in parts per million. Finally, nitrogen describes the annual fertilization rate in tons per hectare per growing season.

Additional units: short-wave radiation [W m-2], precipitation [kg m-2 s-1], temperatures [o C], yield [t ha-1].

Because the dataset is very large, each climate variable is in a separate file to make it easier to work with - you might not want to use all of the variables at first. Each file contains 240 columns reflecting one daily value, from 30 days before sowing until 210 days after, as well as a column called variable with the name of the variable being measured (e.g. pr, tas or rsds), and columns describing the year, latitude, longitude and crop, as well as the unique ID to facilitate joining the datasets.


# Hipothesis

- Nitrogen fertilization rates and soil texture remain constant across longitude and latitude over the years.
- CO2 concentration is constant per year but increases over the years
