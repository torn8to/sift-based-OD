import math

# Depth first search to determine clusters
def dfs(i, set, dict, pose_set, valid_bins):
    if i in set:
        return pose_set
    set.add(i)
    pose_set.append(valid_bins[i])
    for nb in dict[i]:
        pose_set = dfs(nb, set, dict, pose_set, valid_bins)
    return pose_set


def group_position(valid_bins):
    pose_cluster = []
    pose_cluster_dict = {}
    # Form a proximity list for each valid bin based on a threshold
    for b in range(len(valid_bins)):
        if b not in pose_cluster_dict.keys():
            pose_cluster_dict[b] = []
        for a in range(b):
            x1, y1 = valid_bins[a].centroid
            x2, y2 = valid_bins[b].centroid
            if a not in pose_cluster_dict.keys():
                pose_cluster_dict[a] = []

            if abs(x1 - x2) <= valid_bins[a].img_size[0] * valid_bins[a].scale / 4 and abs(y1- y2) <= valid_bins[a].img_size[1] * valid_bins[a].scale / 4:
                if abs(x1 - x2) <= valid_bins[b].img_size[0] * valid_bins[b].scale / 4 and abs(y1- y2) <= valid_bins[b].img_size[1] * valid_bins[b].scale / 4:
                    pose_cluster_dict[b].append(a)
                    pose_cluster_dict[a].append(b)
    seen = set()
    # Apply Depth First search to determine groups of nearby points
    for i in range(len(valid_bins)):
        if i not in seen:
            pose_cluster.append([])
            pose_cluster[len(pose_cluster) - 1] = dfs(i, seen, pose_cluster_dict, pose_cluster[len(pose_cluster) - 1], valid_bins)
    return pose_cluster

def group_orientation(pose_cluster):
    orientation_cluster = []
    for i in range(len(pose_cluster)):
        orient_dict = {}
        # Form a proximity list of orientations for each of the position clusters
        for b in range(len(pose_cluster[i])):
            if b not in orient_dict.keys():
                orient_dict[b] = []
            for a in range(b):
                theta1 = pose_cluster[i][a].angle
                theta2 = pose_cluster[i][b].angle
                if a not in orient_dict.keys():
                    orient_dict[a] = []
                if abs(math.degrees(theta1 - theta2)) <= 1:
                    orient_dict[a].append(b)
                    orient_dict[b].append(a)
        seen = set()
        orientation_cluster.append([])
        # Apply dfs on the proximity list and form orientation clusters for each pose cluster
        for k in range(len(pose_cluster[i])):
            if k not in seen:
                orientation_cluster[i].append([])
                last = len(orientation_cluster[i]) - 1
                orientation_cluster[i][last] = dfs(k, seen, orient_dict, orientation_cluster[i][last], pose_cluster[i])
    return orientation_cluster

def find_max_orientation(orientation_cluster):
    final_orientation_list = []
    for i in range(len(orientation_cluster)):
        max_vote = 0
        final_orientation = 0
        # Determine the maximum votes for orientation cluster of each pose cluster
        for j in range(len(orientation_cluster[i])):
            if len(orientation_cluster[i][j]) >= max_vote:
                # for cluster in orientation_cluster[i][j]:
                #     print(math.degrees(cluster.angle))
                max_vote = len(orientation_cluster[i][j])
                final_orientation = 0  
                # Represent the mean of max voted orientation cluster as the final orientation for the pose cluster
                for k in range(len(orientation_cluster[i][j])):
                    final_orientation += orientation_cluster[i][j][k].angle
                final_orientation = final_orientation / len(orientation_cluster[i][j])
        final_orientation_list.append(final_orientation)
    return final_orientation_list



def get_final_pose(pose_cluster, final_orientation_list):
    final_pose = []
    for i, pose in enumerate(pose_cluster):
        mean_centroid_x = 0
        mean_centroid_y = 0
        mean_orientation =final_orientation_list[i]
        mean_scale = 0
        min_area = math.inf
        min_shape = 0, 0
        # Calculate the mean x, y for each pose cluster and mean scale.
        # Also determine the image size for plotting rectangle by selecting the rectangle with minimum area
        for j in range(len(pose_cluster[i])):
            mean_centroid_x += pose_cluster[i][j].centroid[0]
            mean_centroid_y += pose_cluster[i][j].centroid[1]
            mean_scale += pose_cluster[i][j].scale
            temp_width = pose_cluster[i][j].img_size[0] * pose_cluster[i][j].scale
            temp_height = pose_cluster[i][j].img_size[1] * pose_cluster[i][j].scale
            temp_area = temp_width * temp_height
            if temp_area < min_area:
                min_area = temp_area
                min_shape = pose_cluster[i][j].img_size
        mean_centroid_x = mean_centroid_x / len(pose_cluster[i])
        mean_centroid_y = mean_centroid_y / len(pose_cluster[i])
        mean_centroid = mean_centroid_x, mean_centroid_y
        mean_scale = mean_scale / len(pose_cluster[i])
        final_pose.append((mean_centroid, mean_orientation, mean_scale, min_shape))
    return final_pose
