import numpy as np

BOARD_LENGTH = 10 # panjang board, mulai dari 0
HOLE_POSITION = 0 # posisi lubang
APPLE_POSITION = 9 # posisi apel
START_POSITION = 2 # posisi mulai

class Environment:
    def __init__(self):
        self.board_length = BOARD_LENGTH
        self.hole_position = HOLE_POSITION
        self.apple_position = APPLE_POSITION
        self.start_position = START_POSITION

    def get_next_state(self, state, action):
        if action == 0:  # gerak ke kiri
            return max(0, state - 1)
        else:  # gerak ke kanan
            return min(self.board_length - 1, state + 1)

    def get_reward(self, state):
        if state == self.hole_position:
            return -100
        elif state == self.apple_position:
            return 100
        else:
            return -1
