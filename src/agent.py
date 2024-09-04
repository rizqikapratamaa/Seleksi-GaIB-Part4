import numpy as np
import random

ALPHA = 0.3  # learning rate
GAMMA = 0.9  # discount factor
EPSILON = 0.3  # exploration rate
EPISODES = 1000

class QTable:
    def __init__(self, state_size, action_size):
        self.q_table = np.zeros((state_size, action_size))

    def update_q_learning(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] += ALPHA * (reward + GAMMA * self.q_table[next_state, best_next_action] - self.q_table[state, action])

    def update_sarsa(self, state, action, reward, next_state, next_action):
        self.q_table[state, action] += ALPHA * (reward + GAMMA * self.q_table[next_state, next_action] - self.q_table[state, action])

    def get_best_action(self, state):
        return np.argmax(self.q_table[state])

    def print_q_table(self):
        print("\nQ-Table:")
        print("        Left     Right")
        for state in range(self.q_table.shape[0]):
            print(f"{state:2d} {self.q_table[state, 0]:.6f} {self.q_table[state, 1]:.6f}")

class QLearning:
    def __init__(self, env):
        self.env = env
        self.q_table = QTable(env.board_length, 2)  # 2 actions: left, right

    def choose_action(self, state):
        # epsilon greedy policy untuk memilih action terbaik
        if random.uniform(0, 1) < EPSILON:
            return random.randint(0, 1)  # random action
        else:
            return self.q_table.get_best_action(state)  # pilih best action

    def train(self):
        state = self.env.start_position
        total_score = 0
        path = [state]
        win_episode = -1
        lose_episode = -1

        for episode in range(EPISODES):
            if total_score >= 500:
                win_episode = episode
                break
            if total_score <= -200:
                lose_episode = episode
                break

            action = self.choose_action(state)
            next_state = self.env.get_next_state(state, action)
            reward = self.env.get_reward(next_state)
            total_score += reward

            self.q_table.update_q_learning(state, action, reward, next_state) # gunakan q-value terbaik dari next_state (off-policy)

            state = next_state
            path.append(state)

            if state == self.env.hole_position or state == self.env.apple_position:
                state = self.env.start_position
                path.append(state)

        return total_score, path, win_episode, lose_episode

class SARSA:
    def __init__(self, env):
        self.env = env
        self.q_table = QTable(env.board_length, 2)  # 2 actions: left, right

    def choose_action(self, state):
        # epsilon greedy policy untuk memilih action terbaik
        if random.uniform(0, 1) < EPSILON:
            return random.randint(0, 1)  # random action
        else:
            return self.q_table.get_best_action(state)  # choose best action

    def train(self):
        state = self.env.start_position
        total_score = 0
        path = [state]
        win_episode = -1
        lose_episode = -1

        action = self.choose_action(state)

        for episode in range(EPISODES):
            if total_score >= 500:
                win_episode = episode
                break
            if total_score <= -200:
                lose_episode = episode
                break

            next_state = self.env.get_next_state(state, action)
            reward = self.env.get_reward(next_state)
            total_score += reward

            next_action = self.choose_action(next_state)

            self.q_table.update_sarsa(state, action, reward, next_state, next_action) # gunakan q-value dari next_action yang sebenarnya dipilih pada next_state

            state = next_state
            action = next_action # perbarui action dengan next_action (on-policy)
            path.append(state)

            if state == self.env.hole_position or state == self.env.apple_position:
                state = self.env.start_position
                action = self.choose_action(state) # pilih action baru setelah reset
                path.append(state)

        return total_score, path, win_episode, lose_episode
