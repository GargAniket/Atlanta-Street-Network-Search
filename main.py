"""
Aniket Garg
ECE 3251 HW #2 Submission Script
"""

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import pandas as pd
import math

# read given csv files
nodes_df = pd.read_csv('nodes.csv')
edges_df = pd.read_csv('edges.csv')

# create nodes dictionary
nodes = {}
for i, row in nodes_df.iterrows():
    nodes[int(row['osmid'])] = (row['y'], row['x'])

# create edges dictionary
edges = {}
for i, row in edges_df.iterrows():
    edges[frozenset([int(row['u']), int(row['v'])])] = row['length']

# create neighbors dictionary
neighbors = {}
for node in nodes_df['osmid'].values.tolist():
    neighbors[node] = set()  # create an empty set for each unique node, add all neighbors to it
    for u, v in zip(edges_df['u'].values.tolist(), edges_df['v'].values.tolist()):
        if node == u:
            neighbors[node].add(int(v))
        if node == v:
            neighbors[node].add(int(u))


# heavyside function to find straight line distance between two points
def straight_line_distance(start, end):
    """takes in coordinate degree values and returns the straight-line distance between them in km"""
    (lat1, long1) = nodes[start]
    (lat2, long2) = nodes[end]

    lat1 = lat1 * math.pi / 180
    lat2 = lat2 * math.pi / 180
    long1 = long1 * math.pi / 180
    long2 = long2 * math.pi / 180
    r = 6373
    h = ((math.sin((lat2 - lat1) / 2)) ** 2) + \
        (math.cos(lat1) * math.cos(lat2)) * \
        ((math.sin((long2 - long1) / 2)) ** 2)
    d = 2 * r * math.asin(math.sqrt(h))
    return abs(d)


# use to get solution path
def reconstruct_path(cameFrom, current):
    total_path = [current]
    while current in cameFrom.keys():
        current = cameFrom[current]
        total_path.insert(0, current)
    return total_path


# used to return node with smallest fScore in openSet
def min_node(openSet, fScore):
    min_node = -1
    min_score = math.inf
    for node in openSet:
        if fScore[node] < min_score:
            min_node = node
            min_score = fScore[node]
    return min_node


def find_a_star_path(start, end):
    # NOTE: use straight_line_distance as heuristic function

    openSet = {start}  # openSet: records visited nodes
    cameFrom = {}  # cameFrom[n]: records parent node of current node
    gScore = {}  # gScore[n]: current shortest path to node n
    fScore = {}  # fScore[n] = gScore[n] + heuristic(n, end): measures estimated distance to end through n

    # initialize all non-start values with infinity
    for n in nodes_df['osmid'].values.tolist():
        gScore[n] = math.inf
        fScore[n] = math.inf
    gScore[start] = 0
    fScore[start] = straight_line_distance(start, end)

    while len(openSet) != 0:
        current = min_node(openSet, fScore)  # select current node as one with the lowest estimated path to end
        if current == -1:
            # failure case
            return "error in finding min node"
        if current == end:
            # if current node is same as end node, then entire path has been constructed
            return reconstruct_path(cameFrom, current)
        openSet.remove(current)
        for n in neighbors[current]:  # check all neighboring nodes
            # curr_gScore: distance from start to neighboring node n along current path
            curr_gScore = gScore[current] + edges[frozenset([current, n])]
            if curr_gScore < gScore[n]:  # if a path has been found to n that is shorter than what is in gScore
                cameFrom[n] = current
                gScore[n] = curr_gScore
                fScore[n] = curr_gScore + straight_line_distance(n, end)
                if n not in openSet:
                    openSet.add(n)

    # failure case
    return "open set empty but goal never reached"


solution = find_a_star_path(69589641, 69272171)
print(f"nodes path: {solution}")

# get all line segments for the solution path
sol_segs = []
for i in range(len(solution)-1):
    sol_segs.append([(nodes[solution[i]][1], nodes[solution[i]][0]),
                     (nodes[solution[i+1]][1], nodes[solution[i+1]][0])])

# get all line segments from all nodes
segs = []
for node, n in neighbors.items():
    for neighbor in n:
        segs.append([(nodes[node][1], nodes[node][0]),
                     (nodes[neighbor][1], nodes[neighbor][0])])

# create line collections of line segments
# plot all images and line collections (green is street network, red is solution path)
lc1 = LineCollection(segs, colors=['g'], linewidths=[0.25])
lc2 = LineCollection(sol_segs, colors=['r'])

fig = plt.figure()
ax = fig.add_subplot()

ax.add_collection(lc1)
ax.autoscale()
ax.add_collection(lc2)
ax.autoscale()
ax.imshow(plt.imread('atlanta.png'), extent=[-84.6536, -84.2215, 33.6407, 33.9060])
ax.autoscale()

plt.title("A* Calculated Path Between Two Points")
plt.show()
