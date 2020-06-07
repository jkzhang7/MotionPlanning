import numpy as np
import collections


class AStarPlanner(object):
    def __init__(self, planning_env, epsilon):
        self.env = planning_env
        self.nodes = {}
        self.epsilon = epsilon
        self.visited = np.zeros(self.env.map.shape)

    def Plan(self, start_config, goal_config):
        # TODO: YOUR IMPLEMENTATION HERE
        plan = []
        # plan.append(start_config)
        state_count = 0

        # Create open set and closed set
        open_set = set()  # open set, pos - priority
        close_set = set()
        dires = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        open_set.add((start_config[0, 0], start_config[1, 0]))
        open_set_f = dict()
        open_set_f[(start_config[0, 0], start_config[1, 0])] = self.env.h(start_config)
        self.nodes[(start_config[0, 0], start_config[1, 0])] = [start_config]
        cost = 0
        while len(open_set):
            cur_config = sorted(open_set_f.items(), key=lambda f: f[1])[0][0]
            state_count += 1
            if cur_config[0] == goal_config[0, 0] and cur_config[1] == goal_config[1, 0]:
                plan += self.nodes[cur_config]
                break
            open_set.remove(cur_config)
            close_set.add(cur_config)
            del open_set_f[cur_config]
            cur_config_g_x = cur_config[0] - start_config[0, 0]
            cur_config_g_y = cur_config[1] - start_config[1, 0]
            cur_config_g = (cur_config_g_x ** 2 + cur_config_g_y ** 2) ** 0.5
            for dx, dy in dires:
                tmp_x = cur_config[0] + dx
                tmp_y = cur_config[1] + dy
                if not self.env.state_validity_checker(np.array([[tmp_x], [tmp_y]])):
                    continue
                if not self.env.edge_validity_checker(np.array([[tmp_x], [tmp_y]]),
                                                      np.array([[cur_config[0]], [cur_config[1]]])):
                    continue
                if (tmp_x, tmp_y) in close_set:
                    continue
                tmp_config_g_x = tmp_x - start_config[0, 0]
                tmp_config_g_y = tmp_y - start_config[1, 0]
                tmp_config_g = (tmp_config_g_x ** 2 + tmp_config_g_y ** 2) ** 0.5
                if (tmp_x, tmp_y) not in open_set or tmp_config_g < cur_config_g:
                    self.nodes[(tmp_x, tmp_y)] = self.nodes[cur_config] + [[cur_config[0], cur_config[1]]]
                    h_score = self.epsilon * self.env.h(np.array([[tmp_x], [tmp_y]]))
                    open_set_f[(tmp_x, tmp_y)] = h_score + tmp_config_g
                    open_set.add((tmp_x, tmp_y))
        plan.append([goal_config[0, 0], goal_config[1, 0]])
        plan[0] = ([start_config[0, 0], start_config[1, 0]])
        for i in range(len(plan)-1):
            cost += ((plan[i+1][0] - plan[i][0]) ** 2 + (plan[i+1][1] - plan[i][1]) ** 2 ) ** 0.5
        for i in range(len(plan)):
            plan[i] = np.array([[plan[i][0]], [plan[i][1]]])
        print("States Expanded: %d" % state_count)
        # cost = open_set_f[(goal_config[0, 0], goal_config[1, 0])]
        print("Cost: %f" % cost)

        return np.concatenate(plan, axis=1)
