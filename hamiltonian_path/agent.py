from game import SnakeGameAI, Direction, Point, delta_to_dir
from path_search import PathSearch
from utils import plot


class Agent:
    def __init__(self, game) -> None:
        self.game = game
        self.n_games = 0
        self.path_search = PathSearch()
    
    def play(self):
        game_over = False
        score = 0
        self.game.reset()

        while not game_over:
            curr_state = self._get_game_state()
            action = self._get_action(curr_state)
            game_over, score = self.game.play_step(action)

        self.n_games += 1
        return self.n_games, score
    
    def _get_game_state(self):
        return self.game.w, self.game.h, self.game.snake, self.game.direction, self.game.food

    def _get_action(self, state):

        self.path_search.initialize(state)

        next_pt = self.path_search.get_path(state)
        head = state[2][0]
        delta = Point(next_pt.x-head.x, next_pt.y-head.y)
        action = delta_to_dir(delta)

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
        print(f"Game: {n_games:<4} Score: {score:<4} Mean score: {mean_score:<6.3} Record: {record:<4}")

if __name__ == "__main__":
    play()
