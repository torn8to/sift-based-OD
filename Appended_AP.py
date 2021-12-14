from AffineParameters import *

appended_affine_parameters = []

# Calculates the Affine Parameters for all the bins
for i in range(0, count):
    appended_affine_parameters.append(AffineParameters(i))

print("done")
