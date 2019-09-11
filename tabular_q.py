# To do a Tic Tac Toe AI using reinforcement learning, we model the game as a Markov Decision Process (MDP)
# In the Markov Decision Process, we have four important variables 
#
# First, we have the state set, which contains all of the possible boards when it is that player's turn
# Thus, the X player who goes first will have a different state set than the O player who goes second
#
# Second, we have the action set, which contains all of the possible moves a player can make at a certain state
# For the same reason, the X and O players will have different sets
#
# Third, we have the probability function. It defines how likely a player can move from a state S_1 to a state S_2 by taking
# an action a. The probability function is implicitly weighted in the Q-learning process since the updated weights are based
# on what is going to be played next
#
# Fourth, we need rewards, which will simply be 1 if win, -1 if loss, 0.5 for a tie, and 0 reward if the game is still in
# progress
#
# We also need a policy function that maps from each state to the action set. The goal is to find a policy that maximizes
# the future rewards
#
# We learn the optimal policy function using the TD(0) method. Each state begins with a value function of 0. For each state 
# that the game visits, we use the following rule (which is a weighted average of the old value and the new state):
# 
#  v(s) = v(s) + alpha * (v(s') + R-v(s))
#
# In this case, v(s) is the previous state (state to update), s' is the current state, R is the reward, and alpha is the 
# learning rate. 
#
# In the simple case, assume that the other opponent is making random moves 
import random
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class TicTacToe():
    
    def __init__(self):
        self.state = '---------'

    # Resets the game
    def reset_game(self):
        self.state = '---------'

    # Check if game over
    def game_over(self):

        # Check if there are any possible moves
        if '-' in self.state:
            return None
        
        # Each list corresponds to the values to check to see if a winner is there
        checks = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7] , [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        for check in checks:
            # Check to see if the strings have a winner
            test = self.state[check[0]] + self.state[check[1]] + self.state[check[2]]
            if test == 'XXX':
                return 'X'
            elif test == 'OOO':
                return 'O'
        
        # Return tie in the last case
        return 'T'

    # Does a move given an action (number) and a piece
    def play_move(self, move, ID):

        # Return false if illegal move 
        if self.state[move] != '-':
            return False

        # Plays a move depending on what number is fed in
        self.state = self.state[:move] + ID + self.state[move+1:]

        # Return true if legal move
        return True

    # Gets the reward via ID
    def get_reward(self, result, ID):

        # No reward for a non-game ending state
        if result == 'T':
            return 0.5
        elif result == ID:
            return 1
        else:
            return 0

    # Trains a bot against a random player
    def train(self, a_1, a_2, batch):

        # Exit if the same ID
        if a_1.ID == a_2.ID:
            return

        # Store total wins
        a_1_wins = 0
        a_1_ties = 0

        # Train on 1000 games per session
        for i in range(batch):

            # Store the result
            result = None

            # Make a turn variable to see who's it is
            turn = 'X'

            # Each game is at most 9 turns
            for i in range(9):

                # Play moves
                if a_1.ID == turn:

                    # Get move and play move
                    move = a_1.get_move(self.state)
                    self.play_move(move, a_1.ID)

                    # Flip turn
                    turn = 'O'

                    # Check game over
                    result = self.game_over()
                    if (result != None):
                        break

                else:
                    # Do random move for other player
                    move = a_2.get_move(self.state)
                    self.play_move(move, a_2.ID)

                    # Flip turn
                    turn = 'X'

                    # Check game over
                    result = self.game_over()
                    if (result != None):
                        break

            # Update if game over
            if result != None:
                a_1.calculate(self.get_reward(result, a_1.ID))
                a_2.calculate(self.get_reward(result, a_2.ID))
                a_1.reset()
                a_2.reset()
                self.reset_game()

                # Add one to wins if won
                if result == a_1.ID:
                    a_1_wins += 1
                elif result == 'T':
                    a_1_ties += 1

        return (a_1_wins, a_1_ties)


# Superclass for aribtrary agent
class Agent:

    def __init__(self, ID, name):
        self.ID = ID
        self.name = name

    # Called to get the next move
    @abstractmethod
    def get_move(self, state):
        pass

    # Called at the end of each game to calculate parameters
    @abstractmethod
    def calculate(self, result):
        pass

    # Called to reset the agent
    @abstractmethod
    def reset(self):
        pass

class Random(Agent):

    def __init__(self, ID):
        # Call superconstructor
        super().__init__(ID, 'Random Agent')

    def get_move(self, state):
        return random.choice([i for i in range(9) if state[i] == '-'])

    def calculate(self, result):
        return

    def reset(self):
        return

class Neural_Q_Agent(Agent):

    def __init__(self, ID):
        # Call superconstructor
        super().__init__(ID, 'Neural Q Agent')

    def get_move(self, state):
        return random.choice([i for i in range(9) if state[i] == '-'])

    def calculate(self, result):
        return

    def reset(self):
        return

class Tabular_Q_Agent(Agent):

    def __init__(self, ID, random=False): 

        # Call superconstructor
        super().__init__(ID, 'Tabular Q Agent')

        # State sets, stored as dictionary
        self.states = {}
        self.alpha = 0.9
        self.discount = 0.96
        self.init = 0.6
        self.recorded_states = []

        # For gathering data
        self.gather_data = False
        self.training = []
        self.labels = []

    # For a new game
    def reset(self):
        self.recorded_states = []

    # Calculate stuff for training
    def neural_net_calc(self, result):

        self.recorded_states.reverse()

        # Flag for the first time
        first = True
        maximum = 0

        # For the first case, we just use the reward
        for state in self.recorded_states:

            # Test these three characters
            text = ['X', 'O', '-']
            res = []
            for char in text:
                if state[i] == char:
                    res.training.append(1)
                else:
                    res.training.append(0)

            # Add the list to the training data 
            self.training.append(res)

            # Do special update if first
            if first:
                # Set the win condition to 1
                self.states[state[0]][state[1]] = reward
                first = False
            else:
                if self.gather_data:

                self.states[state[0]][state[1]] = (1 - self.alpha) * self.states[state[0]][state[1]] + self.alpha * \
                                                  self.discount * maximum

            # Store the previous max
            maximum = max(self.states[state[0]])


    # Updates the policy
    def calculate(self, reward):

        # Skip this if we're just gathering data
        if self.gather_data:
            return

        # Iterate through the states
        self.recorded_states.reverse()

        # Flag for the first time
        first = True
        maximum = 0

        # For the first case, we just use the reward
        for state in self.recorded_states:

            # Do special update if first
            if first:
                # Set the win condition to 1
                self.states[state[0]][state[1]] = reward
                first = False
            else:
                if self.gather_data:

                self.states[state[0]][state[1]] = (1 - self.alpha) * self.states[state[0]][state[1]] + self.alpha * \
                                                  self.discount * maximum

            # Store the previous max
            maximum = max(self.states[state[0]])

    # Get q-value list for a key with lazy initialization
    def get_q(self, state):
        
        # If the key is there return it
        if state in self.states:
            return self.states[state]
        # Otherwise initialize it and set illegal moves to -Inf
        else:
            res = []
            # Initialize the array with -inf for values that are illegal
            for i in range(9):
                if state[i] == '-':
                    res.append(self.init)
                else:
                    res.append(float('-inf'))

            self.states[state] = res
            return res

    # Generates possible states to move to
    def generate_moves(self, state, piece):
        
        # Create a list to store
        moves = []
        
        # Iterate through and add 'X' or 'O' wherever possible
        for i in range(9):
            if state[i] == '-':
                moves.append(state[:i] + piece + state[i+1:])

        return moves

    # Get an optimal move
    def get_move(self, state):


        # Get the q-values for this 
        q_vals = self.get_q(state)

        # Get location of all max values and select a random one
        max_q = max(q_vals)
        move = random.choice([i for i, j in enumerate(q_vals) if j == max_q])

        # Record a tuple of board state and index of move
        self.recorded_states.append((state, move))

        return move


# Initialize environment
p1 = Tabular_Q_Agent('X')
p2 = Tabular_Q_Agent('O')
t = TicTacToe()

# Store results
wins = []
ties = []
losses = []
count = []
batch = 100
for i in range (500):
    print(i)
    res = t.train(p1, p2, batch)
    wins.append(res[0] / batch)
    ties.append(res[1] / batch)
    losses.append((batch - res[0] - res[1]) / batch)
    count.append(batch * i)


# Print stuff for matplotlib
plt.ylabel('Outcomes (%)')
plt.xlabel('Number of Games')
plt.title('P1: ({}) vs. P2: ({})'.format(p1.name, p2.name))

plt.plot(count, wins, 'b-', label='P1 Win')
plt.plot(count, losses, 'r-', label='P2 Win')
plt.plot(count, ties, 'g-', label='Tie')
plt.legend(loc='best')
plt.show()
