import pygame
import random
from enum import Enum
from collections import namedtuple

GREEN = (54, 138, 17)
DARK_GREEN = (58, 109, 13)
VIOLET = (176, 91, 222)
PURPLE = (87, 29, 118)
YELLOW = (222, 182, 0)
RED = (200, 0, 0)
BROWN = (107, 66, 18)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

BS = 40  # BLOCK_SIZE

MIN_SPEED = 10
MAX_SPEED = 1000


Point = namedtuple("Point", "x, y")


def get_manhattan_distance(pa, pb):
    return abs(pb.x - pa.x) + abs(pb.y - pa.y)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


def dir_to_delta(direction):
    dir_to_delta = {
        Direction.RIGHT: Point(1, 0),
        Direction.DOWN: Point(0, 1),
        Direction.LEFT: Point(-1, 0),
        Direction.UP: Point(0, -1),
    }
    return dir_to_delta[direction]


def delta_to_dir(delta):
    delta_to_dir = {
        Point(1, 0): Direction.RIGHT,
        Point(0, 1): Direction.DOWN,
        Point(-1, 0): Direction.LEFT,
        Point(0, -1): Direction.UP,
    }
    return delta_to_dir[delta]


class SnakeGameAI:
    def __init__(self, w=20, h=20):
        pygame.init()
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w * BS, self.h * BS))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.speed = MIN_SPEED
        self.font = pygame.font.Font("arial.ttf", 25)
        self.reset()

    def reset(self):
        self.direction = random.choice(list(Direction))
        delta = dir_to_delta(self.direction)
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - delta.x, self.head.y - delta.y),
            Point(self.head.x - delta.x * 2, self.head.y - delta.y * 2),
        ]

        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        x = random.randint(0, self.w - 1)
        y = random.randint(0, self.h - 1)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_MINUS:
                    self.speed -= 10
                    self.speed = max(MIN_SPEED, self.speed)
                    print(f"Speed down: {self.speed}")
                if event.key == pygame.K_PLUS:
                    self.speed += 10
                    self.speed = min(MAX_SPEED, self.speed)
                    print(f"Speed up: {self.speed}")
                if event.key == pygame.K_q:
                    pygame.quit()
                    exit()

        # 2. move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            if self.score == self.w * self.h - 3:
                game_over = True
                self._update_ui(food=False)

                return game_over, self.score
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(self.speed)

        # 6. return game over and score
        return game_over, self.score

    def _is_collision(self):
        if (
            self.head.x > self.w - 1
            or self.head.x < 0
            or self.head.y > self.h - 1
            or self.head.y < 0
        ):
            return True
        if self.head in self.snake[1:]:
            return True
        return False

    def _update_ui(self, food=True):
        for y in range(0, self.h):
            for x in range(0, self.w):
                if (x + y) % 2 == 0:
                    c = GREEN
                else:
                    c = DARK_GREEN
                pygame.draw.rect(self.display, c, pygame.Rect(x * BS, y * BS, BS, BS))

        last_pt = None
        for i, pt in enumerate(self.snake):
            if last_pt:
                delta_pt = Point(last_pt.x - pt.x, last_pt.y - pt.y)
                color = (
                    50 + (i / len(self.snake)) * (200 - 50),
                    0,
                    200 - (i / len(self.snake)) * (200 - 50),
                )
                if delta_pt == Point(-1, 0):
                    pygame.draw.rect(
                        self.display,
                        color,
                        pygame.Rect(
                            (last_pt.x + 0.5) * BS,
                            last_pt.y * BS + BS // 20,
                            BS,
                            BS * 18 // 20,
                        ),
                    )
                if delta_pt == Point(0, 1):
                    pygame.draw.rect(
                        self.display,
                        color,
                        pygame.Rect(
                            pt.x * BS + BS // 20, (pt.y + 0.5) * BS, BS * 18 // 20, BS
                        ),
                    )
                if delta_pt == Point(1, 0):
                    pygame.draw.rect(
                        self.display,
                        color,
                        pygame.Rect(
                            (pt.x + 0.5) * BS, pt.y * BS + BS // 20, BS, BS * 18 // 20
                        ),
                    )
                if delta_pt == Point(0, -1):
                    pygame.draw.rect(
                        self.display,
                        color,
                        pygame.Rect(
                            last_pt.x * BS + BS // 20,
                            (last_pt.y + 0.5) * BS,
                            BS * 18 // 20,
                            BS,
                        ),
                    )
                color = (
                    50 + ((i - 1) / len(self.snake)) * (200 - 50),
                    0,
                    200 - ((i - 1) / len(self.snake)) * (200 - 50),
                )
                pygame.draw.circle(
                    self.display,
                    color,
                    ((last_pt.x + 0.5) * BS, (last_pt.y + 0.5) * BS),
                    BS * 9 // 20,
                )
            last_pt = pt
        color = (
            50 + ((len(self.snake) - 1) / len(self.snake)) * (200 - 50),
            0,
            200 - ((len(self.snake) - 1) / len(self.snake)) * (200 - 50),
        )
        pygame.draw.circle(
            self.display,
            color,
            ((last_pt.x + 0.5) * BS, (last_pt.y + 0.5) * BS),
            BS * 9 // 20,
        )

        if food:
            pygame.draw.circle(
                self.display,
                RED,
                ((self.food.x + 0.5) * BS, (self.food.y + 0.5) * BS),
                BS * 9 // 20,
            )
            pygame.draw.rect(
                self.display,
                BROWN,
                pygame.Rect(
                    self.food.x * BS + BS // 2 - BS // 20,
                    self.food.y * BS + BS // 5,
                    BS // 10,
                    BS // 3.33,
                ),
            )

        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        if action is not None:
            self.direction = action

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1

        self.head = Point(x, y)
