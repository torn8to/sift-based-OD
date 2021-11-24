import math

def dfs(i, seen, adj, pose_set, unique_best_pose):
    if i in seen:
        return pose_set
    seen.add(i)
    pose_set.append(unique_best_pose[i])
    for nb in adj[i]:
        pose_set = dfs(nb, seen, adj, pose_set, unique_best_pose)
    return pose_set

def group_position(best_pose):
    pose_cluster = []
    pose_cluster_dict = {}
    for b in range(len(best_pose)):
        if b not in pose_cluster_dict.keys():
            pose_cluster_dict[b] = []
        for a in range(b):
            x1, y1 = best_pose[a][0]
            x2, y2 = best_pose[b][0]
            if a not in pose_cluster_dict.keys():
                pose_cluster_dict[a] = []

            if abs(x1 - x2) <= best_pose[a][3][1] * best_pose[a][2] / 4 and abs(y1- y2) <= best_pose[a][3][0] * best_pose[a][2] / 4:
                if abs(x1 - x2) <= best_pose[b][3][1] * best_pose[b][2] / 4 and abs(y1- y2) <= best_pose[b][3][0] * best_pose[b][2] / 4:
                    pose_cluster_dict[b].append(a)
                    pose_cluster_dict[a].append(b)
    seen = set()
    for i in range(len(best_pose)):
        print(best_pose[i])
        if i not in seen:
            pose_cluster.append([])
            pose_cluster[len(pose_cluster) - 1] = dfs(i, seen, pose_cluster_dict, pose_cluster[len(pose_cluster) - 1], best_pose)
    return pose_cluster

def group_orientation(pose_cluster):
    orientation_cluster = []
    for i in range(len(pose_cluster)):
        orient_dict = {}
        for b in range(len(pose_cluster[i])):
            if b not in orient_dict.keys():
                orient_dict[b] = []
            for a in range(b):
                theta1 = pose_cluster[i][a][1]
                theta2 = pose_cluster[i][b][1]
                if a not in orient_dict.keys():
                    orient_dict[a] = []
                if abs(math.degrees(theta1 - theta2)) <= 1:
                    orient_dict[a].append(b)
                    orient_dict[b].append(a)
        seen = set()
        orientation_cluster.append([])
        for k in range(len(pose_cluster[i])):
            # print(best_pose[i])
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
        for j in range(len(orientation_cluster[i])):
            # print(len(orientation_cluster[i]))
            if len(orientation_cluster[i][j]) >= max_vote:
                max_vote = len(orientation_cluster[i][j])
                final_orientation = 0  
                for k in range(len(orientation_cluster[i][j])):
                    final_orientation += orientation_cluster[i][j][k][1]
                final_orientation = final_orientation / len(orientation_cluster[i][j])
                # print(final_orientation)
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
        for j in range(len(pose_cluster[i])):
            # if i == 0:
                # print(pose_cluster[i][j][1], pose_cluster[i][j][3])
            mean_centroid_x += pose_cluster[i][j][0][0]
            mean_centroid_y += pose_cluster[i][j][0][1]
            mean_scale += pose_cluster[i][j][2]
            temp_width = pose_cluster[i][j][3][1] * pose_cluster[i][j][2]
            temp_height = pose_cluster[i][j][3][0] * pose_cluster[i][j][2]
            temp_area = temp_width * temp_height
            if temp_area < min_area:
                min_area = temp_area
                min_shape = pose_cluster[i][j][3]
        mean_centroid_x = mean_centroid_x / len(pose_cluster[i])
        mean_centroid_y = mean_centroid_y / len(pose_cluster[i])
        mean_centroid = mean_centroid_x, mean_centroid_y
        mean_scale = mean_scale / len(pose_cluster[i])
        final_pose.append((mean_centroid, mean_orientation, mean_scale, min_shape))
    return final_pose
