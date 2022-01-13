from BaseAI_3 import BaseAI
from Grid_3 import Grid
from time import perf_counter
from copy import deepcopy
from math import log2
from random import randint


class PlayerAI(BaseAI):
    init_time = None
    max_depth = None
    calc_queue = None
    children_queued = None
    time_elapsed = None
    nodes_explored = None
    avg_time_node = .00089

    def __init__(self, grid=None):
        self.grid = grid
        self.heuristic_val = None

    def heuristic(self):
        max_tile = log2(self.grid.getMaxTile())
        max_in_corner = 0
        try:
            merge_weight = .0052 * (1 + (6 / (11 - max_tile)))
        except ZeroDivisionError:
            merge_weight = .0052 * 7
        monotonicity_weight = .0068 * max_tile  # .0067
        empty_spaces_weight = .0047 * (max_tile / 3)  # .005
        merge = merge_weight * (self.merge_potential_h() + .5 * self.ftr_merge_pot())
        if self.max_in_corner(max_tile):
            max_in_corner = .15 * max_tile
        monotonicity = monotonicity_weight * self.monotonicity_h()
        empty_spaces = empty_spaces_weight * self.empty_spaces_h()
        self.heuristic_val = merge + monotonicity + empty_spaces + max_in_corner + max_tile / 20
        return self.heuristic_val

    def max_in_corner(self, max_tile):
        for pos in self.find_all(2 ** max_tile):
            if pos[0] % 3 == 0 and pos[1] % 3 == 0:
                return True

    def merge_potential_h(self):  # high end = 100
        potential = 0
        i = 2
        while log2(i) <= 11:
            indices = self.find_all(i)
            if len(indices) > 0:
                avg_dist = 0
                for index in indices:
                    for sub_index in indices:
                        avg_dist += abs(index[0] - sub_index[0]) + abs(index[1] - sub_index[1]) - 1
                potential += (avg_dist / len(indices) + 1) * (log2(i))
            i *= 2
        return -potential

    def find_all(self, value):
        indices = []
        for row in range(4):
            for col in range(4):
                if self.grid.map[row][col] == value:
                    indices.append((row, col))
        return indices

    def ftr_merge_pot(self):
        potential = 0
        i = 64
        while log2(i) <= 8:
            indices = self.find_all(i * 2)
            indices_i = self.find_all(i)
            if len(indices) > 0 and len(indices_i) > 0:
                avg_dist = 0
                for index in indices:
                    for sub_index in indices_i:
                        avg_dist += abs(index[0] - sub_index[0]) + abs(index[1] - sub_index[1]) - 2
                potential += (avg_dist / (len(indices) * len(indices_i))) * (log2(i)) * .001
            i *= 2
        return -potential

    def monotonicity_h(self):
        penalty = 0
        penalties = [0, 0]

        for row in range(4):
            map_row = self.grid.map[row]
            self.out_of_place(map_row, penalties)

        penalty += min(penalties)
        penalties = [0, 0]

        for col in range(4):
            map_col = [self.grid.map[i][col] for i in range(4)]
            self.out_of_place(map_col, penalties)

        penalty += min(penalties[0], penalties[1])
        return -penalty

    @staticmethod
    def out_of_place(map_sec, penalties):
        try:
            penalties[0] += log2(max([map_sec[i] for i in range(3) if map_sec[i] > map_sec[i + 1]]))
        except ValueError:
            pass
        try:
            penalties[1] += log2(max(map_sec[i + 1] for i in range(3) if map_sec[i + 1] > map_sec[i]))
        except ValueError:
            pass

    def empty_spaces_h(self):  # high end = 9-12
        count = 0
        for row in range(4):
            for col in range(4):
                if self.grid.map[row][col] == 0:
                    count += 1
        return count ** 2

    def children(self, player, self_depth):
        children = []
        if player:
            for move in self.grid.getAvailableMoves():
                new_grid = deepcopy(self.grid)
                new_grid.move(move)
                children.append(PlayerAI(deepcopy(new_grid)))
            children.sort(key=lambda x: x.heuristic())
        else:
            for pos in self.grid.getAvailableCells():
                if pos[0] % 3 == 0 or pos[1] % 3 == 0:
                    new_grid = deepcopy(self.grid)
                    new_grid.insertTile(pos, 2)
                    children.append(PlayerAI(deepcopy(new_grid)))
            children.sort(key=lambda x: x.heuristic(), reverse=True)
        return children

        # self.enough_time(len(children), deepcopy(self_depth)) else []

    def minimax(self, state, alpha, beta, max_player, self_depth, init_move):
        PlayerAI.time_elapsed = perf_counter() - PlayerAI.init_time
        children = deepcopy(state.children(max_player, deepcopy(self_depth)))

        # if PlayerAI.calc_queue:
            # PlayerAI.children_queued += len(children)

        if self_depth > PlayerAI.max_depth:
            PlayerAI.max_depth = self_depth

        # if len(children) == 0:
        if self_depth >= 5:
            # PlayerAI.calc_queue = False
            return state.heuristic_val, init_move

        if max_player:
            max_choice = (-10, init_move)
            for child in children:
                min_choice = self.minimax(child, deepcopy(alpha), deepcopy(beta), False, self_depth + 1, init_move)
                if min_choice[0] > max_choice[0]:
                    max_choice = deepcopy(min_choice)
                PlayerAI.nodes_explored += 1
                alpha = max(alpha, deepcopy(max_choice[0]))
                if alpha >= beta:
                    break
            return max_choice
        else:
            min_choice = (10, init_move)
            for child in children:
                max_choice = self.minimax(child, deepcopy(alpha), deepcopy(beta), True, self_depth + 1, init_move)
                if max_choice[0] < min_choice[0]:
                    min_choice = deepcopy(max_choice)
                PlayerAI.nodes_explored += 1
                beta = min(beta, deepcopy(min_choice[0]))
                if alpha >= beta:
                    break
            return min_choice

    def enough_time(self, queueing, self_depth):
        #  estimate time to finish what is queued, don't add so much that this goes over .18
        #  using avg time elapsed per node queued, estimate the amount of time queued with the
        #  approximate total number of nodes in play - the number that have been explored
        try:
            avg_branch_factor = PlayerAI.children_queued / PlayerAI.max_depth
            approx_nodes = self.num_nodes(avg_branch_factor, self_depth)
            time_to_finish = self.avg_time_node * (approx_nodes + queueing)
            return False if time_to_finish + PlayerAI.time_elapsed > .2 else True
        except ZeroDivisionError:
            return True

    def num_nodes(self, factor, depth):
        if depth <= 0:
            return 0
        return (factor ** depth) + self.num_nodes(factor, depth - 1)

    def getMove(self, grid):
        self.grid = grid
        PlayerAI.init_time = perf_counter()
        PlayerAI.max_depth = 1
        PlayerAI.calc_queue = True
        PlayerAI.children_queued = 0
        PlayerAI.nodes_explored = 1
        PlayerAI.time_elapsed = 0
        alpha = -10
        beta = 10
        children = []

        for move in self.grid.getAvailableMoves():
            new_grid = deepcopy(self.grid)
            new_grid.move(move)
            children.append([PlayerAI(deepcopy(new_grid)), move])

        children.sort(key=lambda x: x[0].heuristic())
        max_choice = (-10, None)

        for child in children:
            min_choice = self.minimax(child[0], deepcopy(alpha), deepcopy(beta), False, 1, child[1])
            if min_choice[0] > max_choice[0]:
                max_choice = deepcopy(min_choice)
            PlayerAI.nodes_explored += 1
            alpha = max(alpha, deepcopy(max_choice[0]))
            if alpha >= beta:
                break

        print(' found choice in: ', perf_counter() - PlayerAI.init_time, ' sec')
        return max_choice[1]

    def seq_moves(self):
        grids = []
        init_grid = Grid()
        for row in range(4):
            for col in range(4):
                init_grid.map[row][col] = self.grid[row][col]
        for i in range(5):
            copy_grid = deepcopy(init_grid)
            copy_grid.move(randint(0, 3))
            grids.append(copy_grid.map)
        return grids

