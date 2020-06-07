import numpy as np
from RRTTree import RRTTree
import time

class RRTPlanner(object):

    def __init__(self, planning_env, bias = 0.05, eta = 1.0, max_iter = 100000):
        self.env = planning_env         # Map Environment
        self.tree = RRTTree(self.env)
        self.bias = bias                # Goal Bias
        self.max_iter = max_iter        # Max Iterations
        self.eta = eta                  # Distance to extend

    def Plan(self, start_config, goal_config):
        # TODO: YOUR IMPLEMENTATION HERE

        plan_time = time.time()

        # Start with adding the start configuration to the tree.
        self.tree.AddVertex(start_config)

        plan = []
        plan.append(start_config)

        for i in range(self.max_iter):
            x_rand = self.sample(goal_config)
            x_near_id, x_near = self.tree.GetNearestVertex(x_rand)

            # x_near = self.tree.vertices[x_near_id]
            x_new = self.extend(x_near, x_rand)
            x_new = np.array(x_new)
            if (len(x_new) == 0) or (not self.env.state_validity_checker(x_new)):
                continue

            x_new_id = self.tree.AddVertex(x_new)
            self.tree.AddEdge(x_near_id, x_new_id)
            if self.env.compute_distance(x_new, goal_config) < 0.0001:
                end_id = x_new_id
                traj = [x_new]
                while self.tree.edges[end_id] != self.tree.GetRootID():
                    traj.append(self.tree.vertices[end_id])
                    end_id = self.tree.edges[end_id]

                plan += traj[::-1]
                break

        plan.append(goal_config)

        cost = 0
        for i in range(len(plan)-1):
            cost += ((plan[i+1][0] - plan[i][0]) ** 2 + (plan[i+1][1] - plan[i][1]) ** 2 ) ** 0.5
        plan_time = time.time() - plan_time

        print("Cost: %f" % cost)
        print("Planning Time: %ds" % plan_time)

        return np.concatenate(plan, axis=1)

    def extend(self, x_near, x_rand):
        # TODO: YOUR IMPLEMENTATION HERE
        # x_new = []
        dist = self.env.compute_distance(x_near, x_rand)
        if dist == 0:
            return []
        if not self.env.edge_validity_checker(x_near, x_rand):
            return []

        d_new = x_rand - x_near
        d_x = d_new[0, 0] * self.eta
        d_y = d_new[1, 0] * self.eta

        if self.eta <= dist:
            x_new = np.zeros(x_near.shape)
            x_new[0, 0] = x_near[0, 0] + d_x
            x_new[1, 0] = x_near[1, 0] + d_y
            return x_new
        else:
            return x_rand

    def sample(self, goal):
        # Sample random point from map
        if np.random.uniform() < self.bias:
            return goal

        return self.env.sample()
