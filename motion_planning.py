import argparse, re, random, time
import msgpack
from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plt

from planning_utils import a_star, heuristic, create_grid, prune_path, create_graph
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
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
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
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

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

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
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

        # # ## Creating probabilistic roadmap
        # # Create sampler
        # sampler = Sampler(data)
        # polygons = sampler._polygons
        #
        # nodes = sampler.sample(200)
        # print('Created {0} samples'.format(len(nodes)))
        #
        # g = create_graph(nodes, 10, polygons)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))

        # Define starting point on the grid as current location
        grid_start = (-north_offset+int(curLocal[0]), -east_offset+int(curLocal[1]))

        # Set goal as some arbitrary position on the grid
        grid_goal = (-north_offset + 64, -east_offset + 85)
        # grid_goal = (290, 720)
        # grid_goal = None
        # while not grid_goal:
        #     i, j = random.randint(0, grid.shape[0]), random.randint(0, grid.shape[1])
        #     if not grid[i, j]: grid_goal = (i, j)
        # TODO: change this to lat/lon position

        # fig = plt.figure()

        # plt.imshow(grid, cmap='Greys', origin='lower')

        # nmin = np.min(data[:, 0])
        # emin = np.min(data[:, 1])

        # # draw edges
        # for (n1, n2) in g.edges:
        #     plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'black', alpha=0.5)
        #
        # # draw all nodes
        # for n1 in nodes:
        #     plt.scatter(n1[1] - emin, n1[0] - nmin, c='blue')
        #
        # # draw connected nodes
        # for n1 in g.nodes:
        #     plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')

        # Run A* to find a path from start to goal
        # or move to a different search space such as a graph (not done here)
        print('Local Start and Goal: ', grid_start, grid_goal)
        timer = time.time()
        path, _ = a_star(grid, heuristic, grid_start, grid_goal)
        print('Found path of {0} waypoints in {1}'.format(len(path), time.time() - timer))

        # plt.plot(grid_start[1], grid_start[0], 'x')
        # plt.plot(grid_goal[1], grid_goal[0], 'x')
        #
        # if path is not None:
        #     pp = np.array(path)
        #     plt.plot(pp[:, 1], pp[:, 0], 'g')
        #
        # plt.xlabel('NORTH')
        # plt.ylabel('EAST')
        #
        # plt.show()

        # TODO: prune path to minimize number of waypoints
        # TODO (if you're feeling ambitious): Try a different approach altogether!
        timer = time.time()
        path = prune_path(path, grid)
        print('Pruned path to {0} waypoints in {1}'.format(len(path), time.time() - timer))

        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        # self.send_waypoints()

    def start(self):
        # self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        # self.stop_log()


if __name__ == "__main__":
    np.random.seed(123)
    random.seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
