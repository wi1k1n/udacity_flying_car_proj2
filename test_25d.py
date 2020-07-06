import numpy as np, time, random
from planning_utils import a_star, heuristic, create_grid, prune_path
from planning_utils import create_graph, create_grid_and_edges, closest_point, a_star_graph
import networkx as nx
from udacidrone.frame_utils import global_to_local
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 25, 25


TARGET_ALTITUDE = 5
SAFETY_DISTANCE_WIDE = 8
SAFETY_DISTANCE_NARROW = 4
SAFETY_DISTANCE_VORONOI = 4

data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
print('Data loaded')

grid, (north_offset, east_offset), _ = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE_WIDE)
grid_narrow, _, _ = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE_NARROW)
grid_original_shape = grid.shape
print('Grids created')

grid_start = (-north_offset, -east_offset)
# grid_start = (450-north_offset, -158-east_offset)
# grid_goal = (350-north_offset, -190-east_offset)
# grid_goal = (290, 720)

goal =          (-122.392210, 37.793520)
global_home =   (-122.397450, 37.792480, 0)
target_global = global_to_local((goal[0], goal[1], TARGET_ALTITUDE), global_home)
grid_goal = (-north_offset + int(target_global[0]), -east_offset + int(target_global[1]))
print('Path: {0} -> {1}'.format(grid_start, grid_goal))

# plt.figure()
# plt.imshow(grid, origin='lower', cmap='Greys')
# plt.plot((grid_start[1], grid_goal[1]), (grid_start[0], grid_goal[0]), 'g', linewidth=3)
# plt.plot(grid_start[1], grid_start[0], 'go')
# plt.plot(grid_goal[1], grid_goal[0], 'go')
# plt.xlabel('EAST', fontsize=20)
# plt.ylabel('NORTH', fontsize=20)
# plt.show()

# grid_show_padding = 100
# grid_mins = (min(grid_start[0], grid_goal[0]), min(grid_start[1], grid_goal[1]))
# grid_maxs = (max(grid_start[0], grid_goal[0]), max(grid_start[1], grid_goal[1]))
# # grid_bb (top, bottom, left, right)
# grid_bb = (grid_mins[0]-grid_show_padding, grid_maxs[0]+grid_show_padding, grid_mins[1]-grid_show_padding, grid_maxs[1]+grid_show_padding)
# grid = grid[grid_bb[0]:grid_bb[1],grid_bb[2]:grid_bb[3]]
# grid_narrow = grid_narrow[grid_bb[0]:grid_bb[1],grid_bb[2]:grid_bb[3]]
# grid_start = (grid_start[0]-grid_bb[0], grid_start[1]-grid_bb[2])
# grid_goal = (grid_goal[0]-grid_bb[0], grid_goal[1]-grid_bb[2])
# print('Grid shrank from shape {0} to {1}', grid_original_shape, grid.shape)
# print('Path in new grid: {0} -> {1}'.format(grid_start, grid_goal))

###################### A* ######################
# timer = time.time()
# path, _ = a_star(grid, heuristic, grid_start, grid_goal, log_progress_each=5)
# print('Found path of {0} waypoints in {1}'.format(len(path), time.time() - timer))
#
# timer = time.time()
# path = prune_path(path, grid_narrow)
# print('Pruned path to {0} waypoints in {1}'.format(len(path), time.time() - timer))
#
# plt.imshow(grid_narrow, cmap='Greys', origin='lower')
# plt.plot(grid_start[1], grid_start[0], 'x')
# plt.plot(grid_goal[1], grid_goal[0], 'x')
# if path is not None:
# 	pp = np.array(path)
# 	plt.plot(pp[:, 1], pp[:, 0], 'o', color='r')
# 	plt.plot(pp[:, 1], pp[:, 0], 'g')
# plt.xlabel('NORTH')
# plt.ylabel('EAST')
# plt.show()


###################### Voronoi ######################
# Create grid and voronoi edges from data
timer = time.time()
grid, offset, edges = create_grid_and_edges(data, TARGET_ALTITUDE, SAFETY_DISTANCE_VORONOI)
print('Voronoi {0} edges created in {1}s'.format(len(edges), time.time() - timer))

# Construct graph from voronoi edges
G = nx.Graph()
for e in edges:
    G.add_edge(tuple(e[0]), tuple(e[1]), weight=np.linalg.norm(np.array(e[1]) - np.array(e[0])))
print('Graph created')

# Iterate several times for different start/goal locations
for iteration in range(1):
    print('> iteration #{0}'.format(iteration))
    def findRandomFreePosition():
        p = None
        while not p:
            i, j = random.randint(0, grid.shape[0] - 1), random.randint(0, grid.shape[1] - 1)
            if not grid[i, j]: p = (i, j)
        return p

    # grid_start = findRandomFreePosition()
    # grid_goal = findRandomFreePosition()

    # Find closest points in voronoi graph
    graph_start = closest_point(G, grid_start)
    graph_goal = closest_point(G, grid_goal)
    print('Graph path: {0} -> {1}'.format(graph_start, graph_goal))

    # Find path on graph
    timer = time.time()
    path, cost = a_star_graph(G, graph_start, graph_goal)
    if path is None: continue
    print('Found path on graph of {0} waypoints in {1}s'.format(len(path), time.time() - timer))

    # Add to path non-voronoi start&goal waypoints
    path = [(int(p[0]), int(p[1])) for p in path]
    path.insert(0, grid_start)
    path.insert(len(path), grid_goal)

    # Prune path on grid
    timer = time.time()
    pruned_path = prune_path(path, grid_narrow)
    pruned_path = prune_path(pruned_path, grid_narrow)
    print('Pruned path from {0} to {1} waypoints in {2}s'.format(len(path), len(pruned_path), time.time() - timer))
    path = pruned_path

    # Show result
    plt.figure()
    plt.imshow(grid, origin='lower', cmap='Greys')
    for e in edges:
        plt.plot([e[0][1], e[1][1]], [e[0][0], e[1][0]], 'b-')
    if path is not None:
        pp = np.array(path)
        plt.plot(pp[:, 1], pp[:, 0], 'g', linewidth=6)
        plt.plot(pp[:, 1], pp[:, 0], 'o', color='r', markersize=20)
    plt.plot(grid_start[1], grid_start[0], 'gx', markersize=20)
    plt.plot(grid_goal[1], grid_goal[0], 'gx', markersize=20)
    plt.xlabel('EAST', fontsize=20)
    plt.ylabel('NORTH', fontsize=20)
    plt.show()

    # input("Press Enter to continue...")
    print('\n')