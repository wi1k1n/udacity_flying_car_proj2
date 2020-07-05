from enum import Enum
from queue import PriorityQueue
import numpy as np
from shapely.geometry import Polygon, Point, LineString
import networkx as nx
import numpy.linalg as LA
from sklearn.neighbors import KDTree


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


class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """
    LEFT = (0, -1, 1)
    RIGHT = (0, 1, 1)
    UP = (-1, 0, 1)
    DOWN = (1, 0, 1)

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
    valid = [Action.UP, Action.LEFT, Action.RIGHT, Action.DOWN]
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid.remove(Action.UP)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid.remove(Action.DOWN)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid.remove(Action.LEFT)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid.remove(Action.RIGHT)

    return valid

def a_star(grid, h, start, goal):
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set()
    visited.add(start)

    branch = {}
    found = False

    visitedCount = 0
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
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



def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


def bresenham(p1, p2, grid):
    """
    Bresenham algorithm, as implemented in exercises (extended for all possible p1 & p2 locations)
    """
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1

    if np.sign(dx * dy) > 0:
        if dx < 0:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            dx = -dx
            dy = -dy

        d = 0
        i = x1
        j = y1

        while i < x2 and j < y2:
            if grid[int(i), int(j)]: return True
            if d < dx - dy:
                d += dy
                i += 1
            elif d == dx - dy:
                # uncomment these two lines for conservative approach
                # cells.append([i+1, j])
                # cells.append([i, j+1])
                d += dy
                i += 1
                d -= dx
                j += 1
            else:
                d -= dx
                j += 1
    else:
        if dy < 0:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            dx = -dx
            dy = -dy

        d = 0
        i = x1
        j = y1

        while i > x2 and j < y2:
            if grid[int(i), int(j)]: return True
            if d > dx - dy:
                d += dy
                i += 1
            elif d == dx - dy:
                # uncomment these two lines for conservative approach
                # cells.append([i+1, j])
                # cells.append([i, j+1])
                d += dy
                i += 1
                d += dx
                j += 1
            else:
                d += dx
                j += 1
    return False


def prune_path(path, grid):
    """
    Takes path (list of wps) and grid (to check if edge collides with obstacle) and returnes pruned path.
    """
    if path is None: return None

    pruned_path = [path[0]]
    cind = 1
    while cind < len(path):
        lwp = pruned_path[-1]
        cwp = path[cind]
        b = bresenham(lwp, cwp, grid)
        # print(lwp, ' -> ', cwp, ' = ', b)
        if b:
            if path[cind - 1] == path[-1]: pruned_path.append(path[cind])
            else: pruned_path.append(path[cind - 1])
        cind += 1
    if len(pruned_path) == 1: pruned_path.append(path[-1])
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