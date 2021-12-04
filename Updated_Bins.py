from Outliers_Trial import *


number_of_bins = len(valid_bins)

for i in range(0, number_of_bins):
    remove_outliers(i)

print("done")