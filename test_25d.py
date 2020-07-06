import numpy as np, time
from planning_utils import a_star, heuristic, create_grid, prune_path, create_graph, create_grid_and_edges
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 25, 25


TARGET_ALTITUDE = 5
SAFETY_DISTANCE = 8
SAFETY_DISTANCE_NARROW = 4

data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
grid_narrow, _, _ = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE_NARROW)

grid_start = (-north_offset, -east_offset)
grid_goal = (-north_offset + 64, -east_offset + 85)
grid_goal = (290, 720)

grid_show_padding = 100
grid_mins = (min(grid_start[0], grid_goal[0]), min(grid_start[1], grid_goal[1]))
grid_maxs = (max(grid_start[0], grid_goal[0]), max(grid_start[1], grid_goal[1]))
# grid_bb (top, bottom, left, right)
grid_bb = (grid_mins[0]-grid_show_padding, grid_maxs[0]+grid_show_padding, grid_mins[1]-grid_show_padding, grid_maxs[1]+grid_show_padding)
grid = grid[grid_bb[0]:grid_bb[1],grid_bb[2]:grid_bb[3]]
grid_narrow = grid_narrow[grid_bb[0]:grid_bb[1],grid_bb[2]:grid_bb[3]]
grid_start = (grid_start[0]-grid_bb[0], grid_start[1]-grid_bb[2])
grid_goal = (grid_goal[0]-grid_bb[0], grid_goal[1]-grid_bb[2])
print('Local Start and Goal: ', grid_start, grid_goal)

timer = time.time()
path, _ = a_star(grid, heuristic, grid_start, grid_goal, log_progress_each=5)
print('Found path of {0} waypoints in {1}'.format(len(path), time.time() - timer))

timer = time.time()
path = prune_path(path, grid_narrow)
print('Pruned path to {0} waypoints in {1}'.format(len(path), time.time() - timer))

plt.imshow(grid_narrow, cmap='Greys', origin='lower')
plt.plot(grid_start[1], grid_start[0], 'x')
plt.plot(grid_goal[1], grid_goal[0], 'x')
if path is not None:
	pp = np.array(path)
	plt.plot(pp[:, 1], pp[:, 0], 'o', color='r')
	plt.plot(pp[:, 1], pp[:, 0], 'g')
plt.xlabel('NORTH')
plt.ylabel('EAST')
plt.show()