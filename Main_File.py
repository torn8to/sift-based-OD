from Updated_Bins import *
from VisualHelperFunctions import plot_rect

'''
Check the bin with maximum length of keypoint_pairs
'''
max_kpp = len(valid_bins[0].keypoint_pairs)

max_bin_first = 0
max_bin_second = 0

for i in range(0, count):
    if len(valid_bins[i].keypoint_pairs) > max_kpp:
        max_kpp = len(valid_bins[i].keypoint_pairs)
        max_bin_first = i

print("done")
# Affine Parameters are calculated once

# AP second time
num = 50
for _ in range(num):
    appended_affine_parameters = []

    for i in range(0, count):
        appended_affine_parameters.append(AffineParameters(i))


    for i in range(0, number_of_bins):
        remove_outliers(i)


    max_kpp = len(valid_bins[0].keypoint_pairs)
    print("done")

    for i in range(0, count):
        if len(valid_bins[i].keypoint_pairs) > max_kpp:
            max_kpp = len(valid_bins[i].keypoint_pairs)
            max_bin_second = i
    print("done")

print("done")


# Draw a box with Pose of Bin Number = max_bin
'''
Plotting
'''
fig, ax = plt.subplots()
color_count = 0
colors = ['r', 'b', 'g', 'y']

for bin in valid_bins:
    #if bin.votes == valid_bins[max_bin_second].votes:
    print("Most Voted Pose: ", bin.pose, " with ", bin.votes, " votes")
    print("Box Size: ", bin.img_size, " in ", colors[color_count % len(colors)], "\n")
    ax = plot_rect(gray_query, bin, ax, colors[color_count % len(colors)])
    color_count += 1

plt.show()
print("done")
