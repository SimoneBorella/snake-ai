import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from utils import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self, game) -> None:
        self.train = True
        self.model_loaded = False

        self.game = game
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() if exceed max len
        if self.model_loaded:
            self.model = torch.load("./model/model_.pt")
        else:
            self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def play(self):
        game_over = False
        score = 0
        self.game.reset()

        while not game_over:
            curr_state = self._get_game_state()
            action = self._get_action(curr_state)
            reward, game_over, score = self.game.play_step(action)
            if self.train:
                new_state = self._get_game_state()
                self._train_short_memory(curr_state, action, reward, new_state, game_over)
                self._remember(curr_state, action, reward, new_state, game_over)

        if self.train:
            self._train_long_memory()
            self.model.save()
        self.n_games += 1
        return self.n_games, score

    def _get_game_state(self):
        head = self.game.snake[0]
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)

        dir_l = self.game.direction == Direction.LEFT
        dir_r = self.game.direction == Direction.RIGHT
        dir_u = self.game.direction == Direction.UP
        dir_d = self.game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.game._is_collision(point_r))
            or (dir_l and self.game._is_collision(point_l))
            or (dir_u and self.game._is_collision(point_u))
            or (dir_d and self.game._is_collision(point_d)),
            
            # Danger right
            (dir_r and self.game._is_collision(point_d))
            or (dir_l and self.game._is_collision(point_u))
            or (dir_u and self.game._is_collision(point_r))
            or (dir_d and self.game._is_collision(point_l)),

            # Danger left
            (dir_r and self.game._is_collision(point_u))
            or (dir_l and self.game._is_collision(point_d))
            or (dir_u and self.game._is_collision(point_l))
            or (dir_d and self.game._is_collision(point_r)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.game.food.x < self.game.head.x, # food left
            self.game.food.x > self.game.head.x, # food right
            self.game.food.y < self.game.head.y, # food up
            self.game.food.y > self.game.head.y # food down
        ]

        return np.array(state, dtype=int)

    def _remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def _train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory
        states, actions, rewards, next_states, game_overs = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def _train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def _get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        if self.train:
            if self.model_loaded:
                self.epsilon = 30 - self.n_games * 30 / 5000
                self.epsilon = max(self.epsilon, 1.)
            else:
                self.epsilon = 40 - self.n_games * 40 / 200
                self.epsilon = max(self.epsilon, 1.)
        else:
            self.epsilon = 0.

        action = [0, 0, 0]
        if random.randint(1, 100) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action


def play():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = SnakeGameAI()
    agent = Agent(game)

    while True:
        n_games, score = agent.play()

        if score > record:
            record = score
        plot_scores.append(score)
        total_score += score
        mean_score = total_score / n_games
        plot_mean_scores.append(mean_score)

        plot(plot_scores, plot_mean_scores)
        print(f"Game: {n_games:<4} Score: {score:<4} Mean score: {mean_score:<6.3} Record: {record:<4} Epsilon: {agent.epsilon:<4.3}")

if __name__ == "__main__":
    play()
