from random import random
from functools import reduce
from collections import namedtuple
from queue import PriorityQueue, LifoQueue
import numpy as np
from game import Direction, Point, get_manhattan_distance

State = namedtuple("State", ["points", "direction"])

MAX_ITER = 1000

class PathSearch:
    def __init__(self):
        pass

    def get_path(self, game_state):
        path = self._a_star_solve(game_state)
        if path is None:
            path = self._long_path_solve(game_state)
        return path

    def _goal_check(self, current_state, food):
        return current_state.points[-1] == food

    # Cost
    def _c(self, state):
        return len(state.points)-1

    # Heuristic
    def _h(self, state, food):
        return get_manhattan_distance(state.points[-1], food)

    def _f(self, state, food):
        return self._c(state) + self._h(state, food)

    def _a_star_solve(self, game_state):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        dir_to_delta = {}
        dir_to_delta[Direction.RIGHT] = Point(1, 0)
        dir_to_delta[Direction.DOWN] = Point(0, 1)
        dir_to_delta[Direction.LEFT] = Point(-1, 0)
        dir_to_delta[Direction.UP] = Point(0, -1)

        w, h, snake, head_direction, food = game_state
        head = snake[0]

        frontier = PriorityQueue()
        state = State([head], head_direction.value)
        frontier.put((self._f(state, food), state))

        _, current_state = frontier.get()

        visited = []
        ##################################
        # evalueated_points = {}
        ##################################
        
        while not self._goal_check(current_state, food):
            current_pt = current_state.points[-1]
            i = clock_wise.index(Direction(current_state.direction))
            actions = [clock_wise[(i - 1) % 4], clock_wise[i], clock_wise[(i + 1) % 4]]

            for action in actions:
                next_pt = Point(
                    current_pt.x + dir_to_delta[action].x,
                    current_pt.y + dir_to_delta[action].y,
                )
                if (
                    next_pt in visited
                    or next_pt in snake
                    or next_pt.x < 0
                    or next_pt.x >= w
                    or next_pt.y < 0
                    or next_pt.y >= h
                ):
                    continue
                new_state = State(current_state.points + [next_pt], action.value)

                frontier.put((self._f(new_state, food), new_state))
                visited.append(next_pt)
                ####################################
                # evalueated_points[next_pt] = self._f(new_state, food)
                ####################################
            # if len(visited) + len(snake) == w*h:
            #     break
            ################################
            # self.print_state(w, h, snake, food, evalueated_points)
            # input("Continue...")
            ################################

            if frontier.empty():
                break

            _, current_state = frontier.get()

        if self._goal_check(current_state, food):
            current_state.points.pop(0)
            return current_state.points
    
        return None
    

    def _long_path_solve(self, game_state):
        iteration = 0
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        dir_to_delta = {}
        dir_to_delta[Direction.RIGHT] = Point(1, 0)
        dir_to_delta[Direction.DOWN] = Point(0, 1)
        dir_to_delta[Direction.LEFT] = Point(-1, 0)
        dir_to_delta[Direction.UP] = Point(0, -1)

        w, h, snake, head_direction, food = game_state
        head = snake[0]

        frontier = LifoQueue()
        state = State([head], head_direction.value)
        frontier.put(state)

        current_state = frontier.get()

        best_state = None
        
        while True:
            current_pt = current_state.points[-1]
            i = clock_wise.index(Direction(current_state.direction))
            actions = [clock_wise[(i - 1) % 4], clock_wise[i], clock_wise[(i + 1) % 4]]

            for action in actions:
                next_pt = Point(
                    current_pt.x + dir_to_delta[action].x,
                    current_pt.y + dir_to_delta[action].y,
                )
                if (
                    next_pt in current_state.points
                    or next_pt in snake
                    or next_pt.x < 0
                    or next_pt.x >= w
                    or next_pt.y < 0
                    or next_pt.y >= h
                ):
                    continue

                new_state = State(current_state.points + [next_pt], action.value)

                if best_state is None or len(new_state.points) > len(best_state.points):
                    best_state = new_state

                frontier.put(new_state)

            iteration += 1

            if frontier.empty() or iteration>MAX_ITER:
                break

            current_state = frontier.get()

        if best_state:
            best_state.points.pop(0)
            return best_state.points
    
        return None



    def print_state(self, w, h, snake, food, evalueated_points):
        print("     ", end="")
        for i in range(w):
            print(f"{i:2} ", end="")
        print()

        for i in range(h):
            print(f"{i:2} [ ", end="")
            for j in range(w):
                p = Point(j, i)
                if p == food:
                    print("() ", end="")
                elif p in evalueated_points:
                    print(f"{evalueated_points[p]:2} ", end="")
                elif p in snake:
                    print("&& ", end="")
                elif p == food:
                    print("() ", end="")
                else:
                    print("-- ", end="")
            print("]")

