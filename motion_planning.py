import argparse, re, random, time
import sys, math

import msgpack, networkx as nx
from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plt

from planning_utils import a_star, heuristic, create_grid, prune_path, a_star_graph, \
                            create_graph, create_grid_and_edges, closest_point
from sampling import Sampler
import visdom

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()
class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.visdom = False
        self.v = visdom.Visdom()
        if self.v.check_connection():
            print("Connected to visdom")
            self.visdom = True
        else:
            print("Connection to visdom failed!")

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)


        if self.visdom:
            # Plot NE
            ne = np.array([self.local_position[0], self.local_position[1]]).reshape(1, -1)
            self.ne_plot = self.v.scatter(ne, opts=dict(
                title="Local position (north, east)",
                xlabel='North',
                ylabel='East'
            ))
            # Plot D
            d = np.array([self.local_position[2]])
            self.t = 0
            self.d_plot = self.v.line(d, X=np.array([self.t]), opts=dict(
                title="Altitude (meters)",
                xlabel='Timestep',
                ylabel='Down'
            ))
            self.register_callback(MsgID.LOCAL_POSITION, self.update_ne_plot)
            self.register_callback(MsgID.LOCAL_POSITION, self.update_d_plot)
    def update_ne_plot(self):
        ne = np.array([self.local_position[0], self.local_position[1]]).reshape(1, -1)
        self.v.scatter(ne, win=self.ne_plot, update='append')
    def update_d_plot(self):
        d = np.array([self.local_position[2]])
        # update timestep
        self.t += 1
        self.v.line(d, X=np.array([self.t]), win=self.d_plot, update='append')

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()
    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.2:
                if abs(self.local_position[2]) < 0.05:
                    self.disarming_transition()
    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()
    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])
    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        if len(self.waypoints) > 0:
            self.target_position = self.waypoints.pop(0)
            lp, tp = self.local_position, self.target_position
            heading = math.atan2(tp[1]-lp[1], tp[0]-lp[0])
            self.target_position[3] = heading
            print('target position', self.target_position)
            self.cmd_position(*self.target_position)
        else:
            print('no waypoints found. landing')
            self.landing_transition()
    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()
    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()
    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        return
        self.stop()
        self.in_mission = False
    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5
        SAFETY_DISTANCE_WIDE = 8    # safety distance to run A* on
        SAFETY_DISTANCE_NARROW = 5  # safety distance for prunning

        self.target_position[2] = TARGET_ALTITUDE

        # Read lat0, lon0 from colliders into floating point values
        lat0 = lon0 = None
        with open('colliders.csv') as f:
            ns = re.findall("-*\d+\.\d+", f.readline())
            assert len(ns) == 2, "Could not parse lat0 & lon0 from 'colliders.csv'. The file might be broken."
            lat0, lon0 = float(ns[0]), float(ns[1])

        self.set_home_position(lon0, lat0, 0)  # set home position as stated in colliders.csv
        # curLocal - is local position of drone relative home)
        curLocal = global_to_local((self._longitude, self._latitude, self._altitude), self.global_home)
        
        print('global home {0}'.format(self.global_home.tolist()))
        print('position {0}'.format(self.global_position.tolist()))
        print('local position {0}'.format(self.local_position.tolist()))

        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        timer = time.time()
        grid, offset, edges = create_grid_and_edges(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print('Voronoi {0} edges created in {1}s'.format(len(edges), time.time() - timer))
        print('Offsets: north = {0} east = {1}'.format(offset[0], offset[1]))

        G = nx.Graph()
        for e in edges:
            G.add_edge(tuple(e[0]), tuple(e[1]), weight=np.linalg.norm(np.array(e[1]) - np.array(e[0])))
        print('Graph created')

        # Define starting point on the grid as current location
        grid_start = (-offset[0] + int(curLocal[0]), -offset[1] + int(curLocal[1]))

        # Set goal as some arbitrary position on the grid
        # grid_goal = (-offset[0] + 64, -offset[1] + 85)
        # grid_goal = (290, 720)
        grid_goal = None
        while not grid_goal:
            i, j = random.randint(0, grid.shape[0]-1), random.randint(0, grid.shape[1]-1)
            if not grid[i, j]: grid_goal = (i, j)
        # TODO: change this to lat/lon position
        print('Path: {0} -> {1}'.format(grid_start, grid_goal))

        graph_start = closest_point(G, grid_start)
        graph_goal = closest_point(G, grid_goal)
        print('Graph path: {0} -> {1}'.format((graph_start[0]-offset[0], graph_start[1]-offset[1]), (graph_goal[0]-offset[0], graph_goal[1]-offset[1])))

        timer = time.time()
        path, cost = a_star_graph(G, graph_start, graph_goal)
        if path is None:
            print('Could not find path')
            return
        print('Found path on graph of {0} waypoints in {1}s'.format(len(path), time.time() - timer))

        # Add to path exact (non-voronoi) start&goal waypoints
        path = [(int(p[0]), int(p[1])) for p in path]
        path.insert(0, grid_start)
        path.insert(len(path), grid_goal)

        # Prune the path on grid twice
        timer = time.time()
        pruned_path = prune_path(path, grid)
        pruned_path = prune_path(pruned_path, grid)
        print('Pruned path from {0} to {1} waypoints in {2}s'.format(len(path), len(pruned_path), time.time() - timer))
        path = pruned_path

        # Convert path to waypoints
        waypoints = [[p[0] + offset[0], p[1] + offset[1], TARGET_ALTITUDE, 0] for p in path]

        # Set self.waypoints
        self.waypoints = waypoints
        self.send_waypoints()

    def start(self):
        # self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        # self.stop_log()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    parser.add_argument('--seed', type=int, default=random.randint(0, sys.maxsize), help='Seed for random, to make result reproducable')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=600)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
