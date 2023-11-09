import random
from collections import namedtuple
from queue import LifoQueue
from game import Direction, Point, dir_to_delta


State = namedtuple("State", ["points", "direction"])

MAX_ITER = 1000


class PathSearch:
    def __init__(self):
        self.hemiltonian_path = None

    def initialize(self, game_state):
        if self.hemiltonian_path is None:
            # self.hemiltonian_path = self._get_hemiltonian_path_backtrack(game_state)
            self.hemiltonian_path = self._get_hemiltonian_path_prim(game_state)
            # self.print_hemiltonian_path(
            #     game_state, self.hemiltonian_path
            # )

    def get_path(self, game_state):
        # self.print_hemiltonian_path(
        #     game_state, self.hemiltonian_path
        # )
        # input()
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        w, h, snake, head_direction, food = game_state
        head = snake[0]
        head_i = self.hemiltonian_path.index(head)
        food_i = self.hemiltonian_path.index(food)

        i = clock_wise.index(Direction(head_direction))
        actions = [clock_wise[(i - 1) % 4], clock_wise[i], clock_wise[(i + 1) % 4]]

        best_delta = len(self.hemiltonian_path)
        best_i = (head_i + 1) % len(self.hemiltonian_path)

        for action in actions:
            delta = dir_to_delta(action)
            next_pt = Point(head.x + delta.x, head.y + delta.y)

            if (
                next_pt in snake
                or next_pt.x < 0
                or next_pt.x >= w
                or next_pt.y < 0
                or next_pt.y >= h
            ):
                continue

            next_pt_i = self.hemiltonian_path.index(next_pt)

            delta_pt = 0
            if next_pt_i <= food_i:
                delta_pt = food_i - next_pt_i
            else:
                delta_pt = food_i - next_pt_i + len(self.hemiltonian_path)

            delta_snake = 0
            while not self.hemiltonian_path[(food_i + delta_snake) % len(self.hemiltonian_path)] in snake:
                delta_snake += 1

            # if delta_pt + delta_snake < 2*len(snake):
            if delta_snake < len(snake):
                continue

            if delta_pt < best_delta:
                best_delta = delta_pt
                best_i = next_pt_i

        next_pt = self.hemiltonian_path[best_i]

        return next_pt

    def _get_hemiltonian_path_backtrack(self, game_state):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        w, h, snake, head_direction, _ = game_state
        head = snake[0]

        frontier = LifoQueue()
        state = State([head], head_direction.value)
        frontier.put(state)

        current_state = frontier.get()

        while True:
            current_pt = current_state.points[-1]
            i = clock_wise.index(Direction(current_state.direction))
            actions = [clock_wise[(i - 1) % 4], clock_wise[i], clock_wise[(i + 1) % 4]]

            if len(current_state.points) == h * w:
                for action in actions:
                    next_pt = Point(
                        current_pt.x + dir_to_delta(action).x,
                        current_pt.y + dir_to_delta(action).y,
                    )
                    if next_pt == head:
                        return current_state.points

            for action in actions:
                next_pt = Point(
                    current_pt.x + dir_to_delta(action).x,
                    current_pt.y + dir_to_delta(action).y,
                )
                if (
                    next_pt in current_state.points
                    or next_pt.x < 0
                    or next_pt.x >= w
                    or next_pt.y < 0
                    or next_pt.y >= h
                ):
                    continue

                new_state = State(current_state.points + [next_pt], action.value)

                frontier.put(new_state)
            if frontier.empty():
                break
            current_state = frontier.get()
            # self.print_hemiltonian_path(w, h, current_state.points)

        return None

    def _get_hemiltonian_path_prim(self, game_state):
        directions = dict()
        w, h, _, _, _ = game_state

        w //= 2
        h //= 2

        vertices = h * w

        # Creates keys for the directions dictionary
        # Note that the maze has half the width and length of the grid for the hamiltonian cycle
        for i in range(h):
            for j in range(w):
                directions[j, i] = []

        # The initial cell for maze generation is chosen randomly
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        initial_cell = (x, y)

        print(x*2, y*2)

        current_cell = initial_cell

        # Stores all cells that have been visited
        visited = [initial_cell]

        # Contains all neighbouring cells to cells that have been visited
        adjacent_cells = set()

        # Generates walls in grid randomly to create a randomized maze
        while len(visited) != vertices:
            x = current_cell[0]
            y = current_cell[1]

            # Fill set of adjacent cells
            for dx in [-1, 1]:
                if x + dx >= 0 and x + dx < w:
                    adjacent_cells.add((x + dx, y))

            for dy in [-1, 1]:
                if y + dy >= 0 and y + dy < h:
                    adjacent_cells.add((x, y + dy))

            # Generates a wall between two cells in the grid
            while current_cell:
                current_cell = adjacent_cells.pop()

                # The neighbouring cell is disregarded if it is already a wall in the maze
                if current_cell not in visited:
                    # The neighbouring cell is now classified as having been visited
                    visited.append(current_cell)
                    x = current_cell[0]
                    y = current_cell[1]

                    # To generate a wall, a cell adjacent to the current cell must already have been visited
                    # The direction of the wall between cells is stored
                    # The process is simplified by only considering a wall to be to the right or down
                    if (x + 1, y) in visited:
                        directions[x, y] += [Direction.RIGHT]
                    elif (x - 1, y) in visited:
                        directions[x - 1, y] += [Direction.RIGHT]
                    elif (x, y + 1) in visited:
                        directions[x, y] += [Direction.DOWN]
                    elif (x, y - 1) in visited:
                        directions[x, y - 1] += [Direction.DOWN]

                    break

        # Provides the hamiltonian cycle generating algorithm with the direction of the walls to avoid
        return self._hamiltonian_cycle_prim(h, w, directions)

    def _hamiltonian_cycle_prim(self, h, w, directions):
        # The path for the snake is stored in a dictionary
        # The keys are the (x, y) positions in the grid
        # The values are the adjacent (x, y) positions that the snake can travel towards
        hamiltonian_graph = dict()

        # Uses the coordinates of the walls to generate available adjacent cells for each cell
        # Simplified by only considering the right and down directions
        for i in range(h):
            for j in range(w):
                # Finds available adjacent cells if current cell does not lie on an edge of the grid
                if j != w - 1 and i != h - 1 and j != 0 and i != 0:
                    if Direction.RIGHT in directions[j, i]:
                        hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 2, i * 2)]
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [
                            (j * 2 + 2, i * 2 + 1)
                        ]
                    else:
                        hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.DOWN in directions[j, i]:
                        hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2, i * 2 + 2)]
                        if (j * 2 + 1, i * 2 + 1) in hamiltonian_graph:
                            hamiltonian_graph[j * 2 + 1, i * 2 + 1] += [
                                (j * 2 + 1, i * 2 + 2)
                            ]
                        else:
                            hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [
                                (j * 2 + 1, i * 2 + 2)
                            ]
                    else:
                        hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.DOWN not in directions[j, i - 1]:
                        hamiltonian_graph[j * 2, i * 2] = [(j * 2 + 1, i * 2)]
                    if Direction.RIGHT not in directions[j - 1, i]:
                        if (j * 2, i * 2) in hamiltonian_graph:
                            hamiltonian_graph[j * 2, i * 2] += [(j * 2, i * 2 + 1)]
                        else:
                            hamiltonian_graph[j * 2, i * 2] = [(j * 2, i * 2 + 1)]

                # Finds available adjacent cells if current cell is in the bottom right corner
                elif j == w - 1 and i == h - 1:
                    hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                    hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.DOWN not in directions[j, i - 1]:
                        hamiltonian_graph[j * 2, i * 2] = [(j * 2 + 1, i * 2)]
                    elif Direction.RIGHT not in directions[j - 1, i]:
                        hamiltonian_graph[j * 2, i * 2] = [(j * 2, i * 2 + 1)]

                # Finds available adjacent cells if current cell is in the top right corner
                elif j == w - 1 and i == 0:
                    hamiltonian_graph[j * 2, i * 2] = [(j * 2 + 1, i * 2)]
                    hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.DOWN in directions[j, i]:
                        hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2, i * 2 + 2)]
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [
                            (j * 2 + 1, i * 2 + 2)
                        ]
                    else:
                        hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.RIGHT not in directions[j - 1, i]:
                        hamiltonian_graph[j * 2, i * 2] += [(j * 2, i * 2 + 1)]

                # Finds available adjacent cells if current cell is in the right column
                elif j == w - 1:
                    hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.DOWN in directions[j, i]:
                        hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2, i * 2 + 2)]
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [
                            (j * 2 + 1, i * 2 + 2)
                        ]
                    else:
                        hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.DOWN not in directions[j, i - 1]:
                        hamiltonian_graph[j * 2, i * 2] = [(j * 2 + 1, i * 2)]
                    if Direction.RIGHT not in directions[j - 1, i]:
                        if (j * 2, i * 2) in hamiltonian_graph:
                            hamiltonian_graph[j * 2, i * 2] += [(j * 2, i * 2 + 1)]
                        else:
                            hamiltonian_graph[j * 2, i * 2] = [(j * 2, i * 2 + 1)]

                # Finds available adjacent cells if current cell is in the top left corner
                elif j == 0 and i == 0:
                    hamiltonian_graph[j * 2, i * 2] = [(j * 2 + 1, i * 2)]
                    hamiltonian_graph[j * 2, i * 2] += [(j * 2, i * 2 + 1)]
                    if Direction.RIGHT in directions[j, i]:
                        hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 2, i * 2)]
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [
                            (j * 2 + 2, i * 2 + 1)
                        ]
                    else:
                        hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.DOWN in directions[j, i]:
                        hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2, i * 2 + 2)]
                        if (j * 2 + 1, i * 2 + 1) in hamiltonian_graph:
                            hamiltonian_graph[j * 2 + 1, i * 2 + 1] += [
                                (j * 2 + 1, i * 2 + 2)
                            ]
                        else:
                            hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [
                                (j * 2 + 1, i * 2 + 2)
                            ]
                    else:
                        hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]

                # Finds available adjacent cells if current cell is in the bottom left corner
                elif j == 0 and i == h - 1:
                    hamiltonian_graph[j * 2, i * 2] = [(j * 2, i * 2 + 1)]
                    hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.RIGHT in directions[j, i]:
                        hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 2, i * 2)]
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [
                            (j * 2 + 2, i * 2 + 1)
                        ]
                    else:
                        hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.DOWN not in directions[j, i - 1]:
                        hamiltonian_graph[j * 2, i * 2] += [(j * 2 + 1, i * 2)]

                # Finds available adjacent cells if current cell is in the left corner
                elif j == 0:
                    hamiltonian_graph[j * 2, i * 2] = [(j * 2, i * 2 + 1)]
                    if Direction.RIGHT in directions[j, i]:
                        hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 2, i * 2)]
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [
                            (j * 2 + 2, i * 2 + 1)
                        ]
                    else:
                        hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.DOWN in directions[j, i]:
                        hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2, i * 2 + 2)]
                        if (j * 2 + 1, i * 2 + 1) in hamiltonian_graph:
                            hamiltonian_graph[j * 2 + 1, i * 2 + 1] += [
                                (j * 2 + 1, i * 2 + 2)
                            ]
                        else:
                            hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [
                                (j * 2 + 1, i * 2 + 2)
                            ]
                    else:
                        hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.DOWN not in directions[j, i - 1]:
                        hamiltonian_graph[j * 2, i * 2] += [(j * 2 + 1, i * 2)]

                # Finds available adjacent cells if current cell is in the top row
                elif i == 0:
                    hamiltonian_graph[j * 2, i * 2] = [(j * 2 + 1, i * 2)]
                    if Direction.RIGHT in directions[j, i]:
                        hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 2, i * 2)]
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [
                            (j * 2 + 2, i * 2 + 1)
                        ]
                    else:
                        hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.DOWN in directions[j, i]:
                        hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2, i * 2 + 2)]
                        if (j * 2 + 1, i * 2 + 1) in hamiltonian_graph:
                            hamiltonian_graph[j * 2 + 1, i * 2 + 1] += [
                                (j * 2 + 1, i * 2 + 2)
                            ]
                        else:
                            hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [
                                (j * 2 + 1, i * 2 + 2)
                            ]
                    else:
                        hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.RIGHT not in directions[j - 1, i]:
                        hamiltonian_graph[j * 2, i * 2] += [(j * 2, i * 2 + 1)]

                # Finds available adjacent cells if current cell is in the bottom row
                else:
                    hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.RIGHT in directions[j, i]:
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [
                            (j * 2 + 2, i * 2 + 1)
                        ]
                        hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 2, i * 2)]
                    else:
                        hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                    if Direction.DOWN not in directions[j, i - 1]:
                        hamiltonian_graph[j * 2, i * 2] = [(j * 2 + 1, i * 2)]
                    if Direction.RIGHT not in directions[j - 1, i]:
                        if (j * 2, i * 2) in hamiltonian_graph:
                            hamiltonian_graph[j * 2, i * 2] += [(j * 2, i * 2 + 1)]
                        else:
                            hamiltonian_graph[j * 2, i * 2] = [(j * 2, i * 2 + 1)]
        # Provides the coordinates of available adjacent cells to generate directions for the snake's movement
        return self._path_generator_prim(hamiltonian_graph, h * w * 4)

    # Generates a path composed of coordinates for the snake to travel along
    def _path_generator_prim(self, graph, cells):
        # The starting position for the path is at cell (0, 0)
        path = [Point(0, 0)]

        previous_cell = path[0]
        previous_direction = None

        # Generates a path that is a hamiltonian cycle by following a set of general laws
        # 1. If the right cell is available, travel to the right
        # 2. If the cell underneath is available, travel down
        # 3. If the left cell is available, travel left
        # 4. If the cell above is available, travel up
        # 5. The current direction cannot oppose the previous direction (e.g. left --> right)
        while len(path) != cells:
            if (
                previous_cell in graph
                and (previous_cell[0] + 1, previous_cell[1]) in graph[previous_cell]
                and previous_direction != "left"
            ):
                path.append(Point(previous_cell[0] + 1, previous_cell[1]))
                previous_cell = (previous_cell[0] + 1, previous_cell[1])
                previous_direction = Direction.RIGHT
            elif (
                previous_cell in graph
                and (previous_cell[0], previous_cell[1] + 1) in graph[previous_cell]
                and previous_direction != "up"
            ):
                path.append(Point(previous_cell[0], previous_cell[1] + 1))
                previous_cell = (previous_cell[0], previous_cell[1] + 1)
                previous_direction = Direction.DOWN
            elif (
                (previous_cell[0] - 1, previous_cell[1]) in graph
                and previous_cell in graph[previous_cell[0] - 1, previous_cell[1]]
                and previous_direction != Direction.RIGHT
            ):
                path.append(Point(previous_cell[0] - 1, previous_cell[1]))
                previous_cell = (previous_cell[0] - 1, previous_cell[1])
                previous_direction = "left"
            else:
                path.append(Point(previous_cell[0], previous_cell[1] - 1))
                previous_cell = (previous_cell[0], previous_cell[1] - 1)
                previous_direction = "up"

        # Returns the coordinates of the hamiltonian cycle path
        return path

    def print_hemiltonian_path(self, game_state, path):
        w, h, snake, _, food = game_state

        print("     ", end="")
        for i in range(w):
            print(f"{i:3} ", end="")
        print()

        for i in range(h):
            print(f"{i:3} [ ", end="")
            for j in range(w):
                p = Point(j, i)
                if p in snake:
                    print("&&& ", end="")
                elif p == food:
                    print("OOO ", end="")
                elif p in path:
                    print(f"{path.index(p):3} ", end="")
                else:
                    print("-- ", end="")
            print("]")
