# my_qlearn_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Implementation of Approximate Q Learning Agent
# https://ai.berkeley.edu/reinforcement.html#Q8

import random, time
import contest.util as util
import contest.game as game

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point

#################
# Team creation #
#################
# Constants
NUM_TRAINING = 0
TRAINING = False

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=NUM_TRAINING):
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

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

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
                dist = self.get_maze_distance(self.start, pos2)
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


class OffensiveReflexAgent(CaptureAgent):
    """
    A reflex agent that seeks food.
    """
    def register_initial_state(self, game_state):
        # For training
        self.num_training = NUM_TRAINING

        # Initialize variables for Q-learning
        if TRAINING:
            self.epsilon = 0.1
        else:
            self.epsilon = 0
        self.alpha = 0.2
        self.gamma = 0.9
        # Baseline
        # self.weights = {
        #         'bias': -11.92116534264878,
        #         'nearby_food': -3.1952114781441074,
        #         'nearby_ghosts': -28.16490028913357,
        #         'eaten_food': 20.652138764195286
        #     }
        # A* 
        self.weights = {
                'bias': -11.92116534264878,
                'nearby_food': -3.2306176993680276,
                'nearby_ghosts': -16.6612110039328,
                'eaten_food': 22.136254908743442
            }

        # Initialize agent position
        self.start = game_state.get_agent_position(self.index)
        self.features_extractor = FeaturesExtractor(self)
        self.legal_positions = game_state.get_walls().as_list(False)

        CaptureAgent.register_initial_state(self, game_state)
    
    def get_weights(self):
        return self.weights

    def get_q_value(self, game_state, action):
        """
        Return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        features = self.features_extractor.get_features(game_state, action)
        weights = self.get_weights()
        q_value = 0
        for feature in features:
            q_value += weights[feature] * features[feature]
        return q_value

    def get_value(self, game_state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  
        Return 0.0 if at terminal state.
        """
        # No legal actions
        actions = game_state.get_legal_actions(self.index)
        if len(actions) == 0:
            return 0.0

        # Get best action, given current state
        best_action = self.get_policy(game_state)
        # Get Q value for the best action
        value = self.get_q_value(game_state, best_action)

        return value

    def get_policy(self, game_state):
        """
        Compute the best action to take in a state, given Q values. 
        Return None if at terminal state.
        """
        actions = game_state.get_legal_actions(self.index)

        # No legal actions
        if len(actions) == 0:
            return None

        action_values = {}
        best_q_value = float('-inf')

        for action in actions:
            target_q_value = self.get_q_value(game_state, action)
            action_values[action] = target_q_value
            # Update best_q_value
            if target_q_value > best_q_value:
                best_q_value = target_q_value

            # Get best actions
            best_actions = [k for k, v in action_values.items() if v == best_q_value]

        # tie-breaking
        best_action = random.choice(best_actions)
        return best_action

    def choose_action(self, game_state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.
        """
        # Terminal state
        actions = game_state.get_legal_actions(self.index)
        if len(actions) == 0:
            return None

        # Default Reflex Agent decision
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        # Q-Learning decision
        action = None
        # Training
        if TRAINING:
            for action in actions:
                self.update_weights(game_state, action)
        # Determine whether to exploit or explore based on the epsilon value
        if not util.flip_coin(self.epsilon):
            # exploit
            action = self.get_policy(game_state)
        else:
            # explore randomly
            action = random.choice(actions)
        return action
    
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

    def update_q_value(self, game_state, action, next_game_state):
        """
        The parent class calls this to observe a
        state = action => next_game_state and reward transition.
        You should do your Q-Value update here
        """
        # Compute bellman equation
        reward = self.get_reward(game_state, next_game_state)
        old_value = self.get_q_value(game_state, action)
        next_q_value = self.get_value(next_game_state)

        difference = (reward + self.gamma * next_q_value) - old_value
        return difference
    
    def update_weights(self, game_state, action):
        """
        Update weights based on transition
        """
        next_game_state = self.get_successor(game_state, action)
        features = self.features_extractor.get_features(game_state, action)

        # Compute Q-value difference
        difference = self.update_q_value(game_state, action, next_game_state)

        # Loop through all features and update weights
        for feature in features:
            # Initialize with a default value
            if feature not in self.weights:
                self.weights[feature] = 0 
            # Update the weight
            new_w = self.alpha * difference * features[feature]
            self.weights[feature] += new_w

    def get_reward(self, game_state, next_game_state):
        """
        Compute total reward for a given state transition #TODO update
        """
        penalty = 0
        reward = 0

        agent_position = game_state.get_agent_position(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghost = [position for position in enemies if not position.is_pacman and position.get_position() is not None]
        food = self.get_food(game_state).as_list()
        min_food_dist = min([self.get_maze_distance(agent_position, f) for f in food])

        # Penalty for dying
        min_ghost_dist = 0
        if len(ghost) > 0: # if there are ghosts
            min_ghost_dist = min([self.get_maze_distance(agent_position, g.get_position()) for g in ghost])
        if min_ghost_dist == 1: # if the ghost is one step away
            next_position = next_game_state.get_agent_state(self.index).get_position()
            if next_position == self.start: # agent is eaten
                penalty = -100

        # Incentive to improve score
        if self.get_score(next_game_state) > self.get_score(game_state):
            diff = self.get_score(next_game_state) - self.get_score(game_state)
            # Update reward 
            reward += diff * 20 if self.red else diff * -20

        # Reward for eating food
        if min_food_dist == 1:
            new_food = self.get_food(next_game_state).as_list()
            if len(food) - len(new_food) == 1:
                reward = 10

        return reward + penalty
    
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        CaptureAgent.final(self, state)

        if TRAINING:
            # you might want to print your weights here for debugging
            print("Weights")
            print(self.weights)


class FeaturesExtractor:

    def __init__(self, agent_instance):
        self.agent_instance = agent_instance

    def get_features(self, game_state, action):
        """
        Returns a dict from features to counts
        Usually, the count will just be 1.0 for
        indicator functions.
        """
        # initialize 
        features = util.Counter()
        agent_start = self.agent_instance.start
        agent_state = game_state.get_agent_state(self.agent_instance.index)
        agent_position = game_state.get_agent_position(self.agent_instance.index)
        agent_position_x, agent_position_y = agent_position
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(agent_position_x + dx), int(agent_position_y + dy)

        walls = game_state.get_walls()
        enemies = [game_state.get_agent_state(i) for i in self.agent_instance.get_opponents(game_state)]
        ghosts = [p.get_position() for p in enemies if not p.is_pacman and p.get_position() is not None]

        food = self.agent_instance.get_food(game_state)
        closest_food_dist = self.closest_food((next_x, next_y), food, walls)

        # Bias feature
        features["bias"] = 1.0

        # Nearby ghosts feature
        features["nearby_ghosts"] = sum((next_x, next_y) in Actions.get_legal_neighbors(g, walls) for g in ghosts)

        # Eaten food feature
        if not features["nearby_ghosts"] and food[next_x][next_y]:
            features["eaten_food"] = 1.0
        
        # Nearby food feature
        if closest_food_dist is not None:
            features["nearby_food"] = float(closest_food_dist) / (walls.width * walls.height)
        
        return features

    def closest_food(self, pos, food, walls):
        """
        closest_food -- this is similar to the function that we have
        worked on in the search project; here its all in one place
        """
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.get_legal_neighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
