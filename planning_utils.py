from enum import Enum
from queue import PriorityQueue
import numpy as np
from shapely.geometry import Polygon, Point, LineString
import networkx as nx
import numpy.linalg as LA
from sklearn.neighbors import KDTree
from scipy.spatial import Voronoi, voronoi_plot_2d
import time
from bresenham import bresenham


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)
def create_grid_and_edges(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    along with Voronoi graph edges given obstacle data and the
    drone's altitude.
    """
    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))
    # Initialize an empty list for Voronoi points
    points = []
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1
            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])


    # create a voronoi graph based on
    # location of obstacle centres
    graph = Voronoi(points)
    # check each edge from graph.ridge_vertices for collision
    edges = []
    for v in graph.ridge_vertices:
        p1 = graph.vertices[v[0]]
        p2 = graph.vertices[v[1]]
        hit = bresenham((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), grid)

        # If the edge does not hit on obstacle
        # add it to the list
        if hit is not None and not hit:
            # array to tuple for future graph creation step)
            p1 = (p1[0], p1[1])
            p2 = (p2[0], p2[1])
            edges.append((p1, p2))

    return grid, edges

class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """
    LEFT =      (0, -1, 1)
    LEFTUP =    (-1, -1, np.sqrt(2))
    UP =        (-1, 0, 1)
    UPRIGHT =   (-1, 1, np.sqrt(2))
    RIGHT =     (0, 1, 1)
    RIGHTDOWN = (1, 1, np.sqrt(2))
    DOWN =      (1, 0, 1)
    DOWNLEFT =  (1, -1, np.sqrt(2))

    def __str__(self):
        if self == self.LEFT:
            return '<'
        elif self == self.RIGHT:
            return '>'
        elif self == self.UP:
            return '^'
        elif self == self.DOWN:
            return 'v'

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])
def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid = [Action.LEFT,
             Action.LEFTUP,
             Action.UP,
             Action.UPRIGHT,
             Action.RIGHT,
             Action.RIGHTDOWN,
             Action.DOWN,
             Action.DOWNLEFT]
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if y - 1 < 0 or grid[x, y - 1] == 1: valid.remove(Action.LEFT)
    if x - 1 < 0 or y - 1 < 0 or grid[x - 1, y - 1] == 1: valid.remove(Action.LEFTUP)
    if x - 1 < 0 or grid[x - 1, y] == 1: valid.remove(Action.UP)
    if x - 1 < 0 or y + 1 > m or grid[x - 1, y + 1] == 1: valid.remove(Action.UPRIGHT)
    if y + 1 > m or grid[x, y + 1] == 1: valid.remove(Action.RIGHT)
    if x + 1 > n or y + 1 > m or grid[x + 1, y + 1] == 1: valid.remove(Action.RIGHTDOWN)
    if x + 1 > n or grid[x + 1, y] == 1: valid.remove(Action.DOWN)
    if x + 1 > n or y - 1 < 0 or grid[x + 1, y - 1] == 1: valid.remove(Action.DOWNLEFT)

    return valid
def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))
def a_star(grid, h, start, goal, log_progress_each=None):
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set()
    visited.add(start)

    branch = {}
    found = False

    visitedCount = 0
    startTime = time.time()
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if log_progress_each and time.time() - startTime > log_progress_each:
            print('L2 to goal = {0}. Visited {1} cells.'.format(heuristic(current_node, goal), len(visited)))
            startTime = time.time()
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = action.cost + current_cost  # (action.cost + g)
                queue_cost = branch_cost + heuristic(next_node, goal)  # (action.cost + g + h)

                if next_node not in visited:
                    visited.add(next_node)
                    visitedCount += 1
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')

    return path[::-1], path_cost


def collinearity_int(p1, p2, p3):
    # Calculate the determinant of the matrix using integer arithmetic
    return p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]) == 0

def prune_path(path, grid):
    """
    Takes path (list of wps) and grid (to check if edge collides with obstacle) and returnes pruned path.
    """
    if path is None: return None
    if len(path) < 3:
        print('Path consists of less than 3 waypoint!!')
        return path

    pruned_path = [path[0]]
    cind = 2  # start checking from ndd waypoint (because 0th and 1st form line segment -> no need to check)
    while cind < len(path) - 1:
        pwp = pruned_path[-1]  # last stored waypoint
        cwp = path[cind]  # current waypoint
        collision = False
        for cell in bresenham(pwp[0], pwp[1], cwp[0], cwp[1]):
            if grid[cell[0], cell[1]]:
                collision = True
                break
        if collision:
            pruned_path.append(path[cind - 1])
        cind += 1
    pruned_path.append(path[-1])
    return pruned_path
def can_connect(n1, n2, polygons):
    l = LineString([n1, n2])
    for p in polygons:
        if p.crosses(l) and p.height >= min(n1[2], n2[2]):
            return False
    return True
def create_graph(nodes, k, polygons):
    g = nx.Graph()
    tree = KDTree(nodes)
    for n1 in nodes:
        # for each node connect try to connect to k nearest nodes
        idxs = tree.query([n1], k, return_distance=False)[0]

        for idx in idxs:
            n2 = nodes[idx]
            if n2 == n1:
                continue

            if can_connect(n1, n2, polygons):
                g.add_edge(n1, n2, weight=1)
    return g