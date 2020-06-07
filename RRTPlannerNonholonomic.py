import numpy as np
from RRTTree import RRTTree
import time

class RRTPlannerNonholonomic(object):

    def __init__(self, planning_env, bias=0.05, max_iter=10000, num_control_samples=25):
        self.env = planning_env                 # Car Environment
        self.tree = RRTTree(self.env)
        self.bias = bias                        # Goal Bias
        self.max_iter = max_iter                # Max Iterations
        self.num_control_samples = 25           # Number of controls to sample

    def Plan(self, start_config, goal_config):
        # TODO: YOUR IMPLEMENTATION HERE

        plan_time = time.time()

        # Start with adding the start configuration to the tree.
        self.tree.AddVertex(start_config)

        plan = []
        plan.append(start_config)

        for i in range(self.max_iter):
            # print(i)
            x_rand = self.sample(goal_config)
            if not self.env.state_validity_checker(x_rand):
                continue

            x_near_id, x_near = self.tree.GetNearestVertex(x_rand)
            x_new, delta_t, action_t = self.extend(x_near=x_near, x_rand=x_rand)

            if x_new is None:
                continue

            x_new_id = self.tree.AddVertex(x_new)
            self.tree.AddEdge(x_near_id, x_new_id)

            if self.env.goal_criterion(x_new, goal_config):
                end_id = x_near_id
                traj = [x_new]
                while self.tree.edges[end_id] != self.tree.GetRootID():
                    traj.append(self.tree.vertices[end_id])
                    end_id = self.tree.edges[end_id]
                    plan += traj[::-1]

                break

        plan.append(goal_config)
        cost = 0
        for i in range(len(plan)-1):
            cost += self.env.compute_distance(plan[i], plan[i+1])
        plan_time = time.time() - plan_time

        print("Cost: %f" % cost)
        print("Planning Time: %ds" % plan_time)

        return np.concatenate(plan, axis=1)

    def extend(self, x_near, x_rand):
        """ Extend method for non-holonomic RRT

            Generate n control samples, with n = self.num_control_samples
            Simulate trajectories with these control samples
            Compute the closest closest trajectory and return the resulting state (and cost)
        """
        # TODO: YOUR IMPLEMENTATION HERE
        dist = self.env.compute_distance(x_near, x_rand)
        for i in range(self.num_control_samples):
            linear_vel, steer_angle = self.env.sample_action()
            x_new, delta_t = self.env.simulate_car(x_near, x_rand, linear_vel, steer_angle)
            if x_new is None:
                continue
            d = self.env.compute_distance(x_new, x_rand)
            if d < dist:
                action_t = np.array([linear_vel, steer_angle])
                return x_new, delta_t, action_t
        return None, None, None

    def sample(self, goal):
        # Sample random point from map
        if np.random.uniform() < self.bias:
            return goal

        return self.env.sample()