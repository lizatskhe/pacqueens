# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util
import copy


from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.game import Actions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.walls = game_state.data.layout.walls

    def get_deep_walls(self):
        return copy.deepcopy(self.walls)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        actions = actions[:-1]
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.a_star_search (game_state, self.start, pos2, 1)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

import random

class OffensiveReflexAgent(ReflexCaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.last_score = 0
        self.stuck_counter = 0
        self.last_position = None

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # check if the agent is stuck
        current_score = self.get_score(game_state)
        current_position = game_state.get_agent_state(self.index).get_position()

        if current_score == self.last_score and current_position == self.last_position:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self.last_score = current_score
        self.last_position = current_position

        # if stuck for too long, choose a random action
        if self.stuck_counter > 2:
            self.stuck_counter = 0
            return random.choice(actions)

        return random.choice(best_actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_list = self.get_food(successor).as_list()

        # evaluate food collection or return to start
        if my_state.num_carrying < 4:
            features['successor_score'] = -len(food_list)
        else:
            features['successor_score'] = -self.get_maze_distance(my_pos, self.start)

        # evaluate ghost distance
        ghost_distance = self.get_closest_ghost_distance(game_state, my_pos)
        features['distance_to_ghost'] = ghost_distance if ghost_distance < 3 else 0

        # evaluate capsule distance
        capsules = self.get_capsules(game_state)
        if capsules:
            features['distance_to_capsule'] = min(self.a_star_search(game_state, my_pos, cap, True) for cap in capsules)
        else:
            features['distance_to_capsule'] = 0

        # evaluate food distance
        if food_list:
            features['distance_to_food'] = min(self.get_maze_distance(my_pos, food) for food in food_list)

        # feature to encourage exploration
        features['exploration'] = self.get_exploration_score(my_pos)

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,
            'distance_to_food': -1,
            'distance_to_capsule': -10,
            'distance_to_ghost': 10,
            'exploration': 20
        }

    def get_exploration_score(self, position):
        try:
            self.visited_positions[position] += 1
        except AttributeError:
            self.visited_positions = util.Counter()
            self.visited_positions[position] = 1
        return 1.0 / (self.visited_positions[position] + 1.0)


    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_closest_ghost_distance(self, game_state, my_pos):
        opponents = self.get_opponents(game_state)
        ghost_distances = []
        for opponent in opponents:
            ghost_pos = game_state.get_agent_position(opponent)
            if ghost_pos:
                distance = self.a_star_search(game_state, my_pos, ghost_pos, 0)
                ghost_distances.append(distance)
        return min(ghost_distances) if ghost_distances else 9999

    def a_star_search(self, game_state, start, goal, avoid_ghosts):
        walls = self.get_walls_with_ghosts(game_state) if avoid_ghosts else game_state.get_walls()
        fringe = util.PriorityQueue()
        visited = set()
        fringe.push((start, []), self.get_maze_distance(start, goal))

        while not fringe.is_empty():
            (state, path) = fringe.pop()
            if state == goal:
                return len(path)
            if state not in visited:
                visited.add(state)
                for next_state in Actions.get_legal_neighbors(state, walls):
                    new_path = path + [next_state]
                    priority = len(new_path) + self.get_maze_distance(next_state, goal)
                    fringe.push((next_state, new_path), priority)
        return 0

    def get_walls_with_ghosts(self, game_state):
        walls = game_state.get_walls().copy()
        for opponent in self.get_opponents(game_state):
            ghost_pos = game_state.get_agent_position(opponent)
            if ghost_pos:
                x, y = ghost_pos
                walls[x][y] = True
        return walls

class DefensiveReflexAgent(ReflexCaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.patrol_points = []

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.patrol_points = self.get_patrol_points(game_state)

    def get_patrol_points(self, game_state):
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        mid_x = width // 2
        if self.red:
            mid_x = mid_x - 1
        points = [(mid_x, y) for y in range(1, height) if not game_state.has_wall(mid_x, y)]
        return points

    def a_star_search(self, game_state, start, goal, avoid_pacman=False):
        walls = game_state.get_walls()
        fringe = util.PriorityQueue()
        closed = set()
        fringe.push((start, []), self.get_maze_distance(start, goal))

        while not fringe.is_empty():
            (node, path) = fringe.pop()
            if node == goal:
                return len(path)
            if node not in closed:
                closed.add(node)
                for successor in Actions.get_legal_neighbors(node, walls):
                    if avoid_pacman:
                        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
                        pacmen = [a for a in enemies if a.is_pacman and a.get_position() is not None]
                        if any(self.get_maze_distance(successor, a.get_position()) < 2 for a in pacmen):
                            continue
                    new_path = path + [successor]
                    fringe.push((successor, new_path), 
                                len(new_path) + self.get_maze_distance(successor, goal))
        return None

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.a_star_search(game_state,my_pos, a.get_position(), 0) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            # if no invaders, patrol
            patrol_dist = min([self.get_maze_distance(my_pos, p) for p in self.patrol_points])
            features['patrol_distance'] = patrol_dist

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # distance to the nearest food
        our_food = self.get_food_you_are_defending(successor).as_list()
        if len(our_food) > 0:
            min_food_dist = min([self.get_maze_distance(my_pos, food) for food in our_food])
            features['food_distance'] = min_food_dist

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 
                'stop': -100, 'reverse': -2, 'patrol_distance': -1, 'food_distance': 1}
