from Updated_Bins import *
from VisualHelperFunctions import plot_rect

'''
Check the bin with maximum length of keypoint_pairs
'''
max_kpp = len(valid_bins[0].keypoint_pairs)
print("done")

for i in range(0, count):
    if len(valid_bins[i].keypoint_pairs) > max_kpp:
        max_kpp = len(valid_bins[i].keypoint_pairs)
        max_bin = i

# Draw a box with Pose of Bin Number = max_bin
print("done")


'''
Plotting is pending
'''
#plot_rect(gray_query, valid_bins = PoseBin())
#print("done")