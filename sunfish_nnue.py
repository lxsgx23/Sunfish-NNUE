#!/usr/bin/env python3
from __future__ import print_function

import time, math, pickle, re, sys
import numpy as np
from itertools import count
from collections import namedtuple, defaultdict

#TO DO:
#  There seems to be something wrong when loading the engine in my Chessbase17 interface,
# even if it runs smoothly in cutechess(maybe
#  And it failed to display score mate properly.
#  Whats more,its time control is terrible.

# 简化输入检查函数
def has_input():
    """Check if there is any input available"""
    try:
        if sys.platform == "win32":
            import msvcrt
            return msvcrt.kbhit()
        else:
            import select
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    except:
        return False

# NNUE evaluation setup
model = pickle.load(open("tanh.pickle", "br"))

nn = [np.frombuffer(ar, dtype=np.int8) / 127.0 for ar in model["ars"]]
L0, L1, L2 = 10, 10, 10
layer1 = nn[4].reshape(L2, 2 * L1 - 2)
layer2 = nn[5].reshape(1, L2)
pad = np.pad(nn[0].reshape(8, 8, 6)[::-1], ((2, 2), (1, 1), (0, 0))).reshape(120, 6)
pst = np.einsum("sd,odp->pso", pad, nn[1].reshape(L0, 6, 6))
pst = np.einsum("psd,odc->cpso", pst, nn[3].reshape(L0, L0, 2))
pst = dict(zip("PNBRQKpnbrqk", pst.reshape(12, 120, L0)))
pst["."] = [[0]*L0] * 120

def compute_value(wf, bf):
    """Compute position value from white and black features"""
    # Apply activation function (tanh)
    act = np.tanh
    # Compute hidden layer
    hidden = (layer1[:, :9] @ act(wf[1:])) + (layer1[:, 9:] @ act(bf[1:]))
    # Compute output score
    score = layer2 @ act(hidden)
    # Scale and convert to integer
    return int((score.item() + model["scale"] * (wf[0] - bf[0])) * 360)

def features(board):
    """Compute feature vectors for a given board position"""
    wf = sum(pst[p][i] for i, p in enumerate(board) if p.isalpha())
    bf = sum(pst[p.swapcase()][119 - i] for i, p in enumerate(board) if p.isalpha())
    return wf, bf

# pieces values
PIECE_VALUES = {
    'P': 100, 'N': 305, 'B': 333, 'R': 563, 'Q': 950, 'K': 60000,
    'p': -100, 'n': -305, 'b': -333, 'r': -563, 'q': -950, 'k': -60000
}

def material_value(board):
    """evaluate situation (from white side)"""
    total = 0
    for p in board:
        if p in PIECE_VALUES:
            total += PIECE_VALUES[p]
    return total

# Mate values
MATE = 100000
MATE_LOWER = MATE // 2
MATE_UPPER = MATE * 3 // 2

version = "Sunfish NNUE"
author = "The Sunfish Developers"

###############################################################################
# Global constants
###############################################################################

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    "         \n"  #   0 -  9
    "         \n"  #  10 - 19
    " rnbqkbnr\n"  #  20 - 29
    " pppppppp\n"  #  30 - 39
    " ........\n"  #  40 - 49
    " ........\n"  #  50 - 59
    " ........\n"  #  60 - 69
    " ........\n"  #  70 - 79
    " PPPPPPPP\n"  #  80 - 89
    " RNBQKBNR\n"  #  90 - 99
    "         \n"  # 100 -109
    "         \n"  # 110 -119
)

# Lists of possible moves for each piece type.
N, E, S, W = -10, 1, 10, -1
directions = {
    "P": (N, N+N, N+W, N+E),
    "N": (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    "B": (N+E, S+E, S+W, N+W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N+E, S+E, S+W, N+W),
    "K": (N, E, S, W, N+E, S+E, S+W, N+W)
}

# UCI options with their default values and ranges
options = {
    "Quiescent_Search": {"value": 40, "min": 0, "max": 300, "type": "spin"},
    "Quiescent_Search_A": {"value": 140, "min": 0, "max": 300, "type": "spin"},
    "EVAL_ROUGHNESS": {"value": 15, "min": 0, "max": 50, "type": "spin"}
}

###############################################################################
# Chess logic
###############################################################################


Move = namedtuple("Move", "i j prom")

class Position(namedtuple("Position", "board wf bf score wc bc ep kp material")):
    """A state of a chess game
    board -- a 120 char representation of the board
    wf -- white's feature vector
    bf -- black's feature vector
    score -- the board evaluation
    wc -- the castling rights, [west/queen side, east/king side]
    bc -- the opponent castling rights, [west/king side, east/queen side]
    ep - the en passant square
    kp - the king passant square
    material - material value (white perspective)
    """

    def is_attacked(self, square, by_white):
        board = self.board
        # Check attack by horse
        knight_moves = directions['N']
        for d in knight_moves:
            j = square + d
            if board[j].isspace():
                continue
            if by_white and board[j] == 'N':
                return True
            if not by_white and board[j] == 'n':
                return True

        # Check attack by king
        king_moves = directions['K']
        for d in king_moves:
            j = square + d
            if board[j].isspace():
                continue
            if by_white and board[j] == 'K':
                return True
            if not by_white and board[j] == 'k':
                return True

        # Check attack by pawn
        if by_white:
            for d in [9, 11]:
                j = square + d
                if board[j] == 'P':
                    return True
        else:
            for d in [-11, -9]: 
                j = square + d
                if board[j] == 'p':
                    return True

    # Check attack by r/q
        for d in directions['R']:
            for step in count(1):
                j = square + step * (-d) 
                if board[j].isspace():
                    break
                if board[j] == '.':
                    continue
                if by_white and board[j] in 'RQ':
                    return True
                if not by_white and board[j] in 'rq':
                    return True
                break  

        # Check attack by b/q
        for d in directions['B']:
            for step in count(1):
                j = square + step * (-d)  
                if board[j].isspace():
                    break
                if board[j] == '.':
                    continue
                if by_white and board[j] in 'BQ':
                    return True
                if not by_white and board[j] in 'bq':
                    return True
                break  

        return False
    
    def gen_moves(self):
        for i, p in enumerate(self.board):
            if not p.isupper():
                continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    if q.isspace() or q.isupper():
                        break
                    if p == "P":
                        if d in (N, N + N) and q != ".": 
                            break
                        if d == N + N and (i < A1 + N or self.board[i + N] != "."): 
                            break
                        if (
                            d in (N + W, N + E)
                            and q == "."
                            and j not in (self.ep, self.kp, self.kp - 1, self.kp + 1)
                        ):
                            break
                        if A8 <= j <= H8:
                            for prom in "NBRQ":
                                move = Move(i, j, prom)
                                new_pos = self.move(move)
                                try:
                                    king_square = new_pos.board.index('k')
                                except ValueError:
                                    continue
                                if not new_pos.is_attacked(king_square, by_white=True):
                                    yield move
                            break
                    # Generate moves
                    move = Move(i, j, "")
                    new_pos = self.move(move)
                    try:
                        king_square = new_pos.board.index('k')
                    except ValueError:
                        continue
                    if not new_pos.is_attacked(king_square, by_white=True):
                        yield move
                    if p in "PNK" or q.islower():
                        break
                    # handle castle
                    if i == A1 and self.board[j + E] == "K" and self.wc[0]:
                        move = Move(j + E, j + W, "")
                        new_pos = self.move(move)
                        try:
                            king_square = new_pos.board.index('k')
                        except ValueError:
                            continue
                        if not new_pos.is_attacked(king_square, by_white=True):
                            yield move
                    if i == H1 and self.board[j + W] == "K" and self.wc[1]:
                        move = Move(j + W, j + E, "")
                        new_pos = self.move(move)
                        try:
                            king_square = new_pos.board.index('k')
                        except ValueError:
                            continue
                        if not new_pos.is_attacked(king_square, by_white=True):
                            yield move

    def rotate(self, nullmove=False):
        """Rotates the board, preserving enpassant, unless nullmove"""
        rotated = Position(
            self.board[::-1].swapcase(), 
            self.bf,  # Swap feature vectors
            self.wf,
            -self.score, 
            self.bc, 
            self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0,
            -self.material  
        )
        return rotated

    def move(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        
        # Helper function to update board and features
        def put(board, wf, bf, i, p):
            new_board = board[:i] + p + board[i + 1 :]
            # Update feature vectors
            new_wf = wf.copy()
            new_bf = bf.copy()
            if board[i] != '.':
                # Remove old piece from features
                old_piece = board[i]
                new_wf -= pst[old_piece][i]
                new_bf -= pst[old_piece.swapcase()][119 - i]
            if p != '.':
                # Add new piece to features
                new_wf += pst[p][i]
                new_bf += pst[p.swapcase()][119 - i]
            return new_board, new_wf, new_bf
        
        # Copy current state
        board = self.board
        wf, bf = self.wf.copy(), self.bf.copy()
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        material = self.material  # Initialize the pieces value
        
        # Remove moving piece from start square
        wf -= pst[p][i]
        bf -= pst[p.swapcase()][119 - i]
        
        # Handle captures
        if q.islower():
            wf -= pst[q][j]
            bf -= pst[q.swapcase()][119 - j]
            material -= PIECE_VALUES[q]
        
        # Place moving piece on target square
        new_p = prom if prom else p
        wf += pst[new_p][j]
        bf += pst[new_p.swapcase()][119 - j]
        
        
        material -= PIECE_VALUES[p]
        material += PIECE_VALUES[new_p]
        
        # Update board
        board = board[:i] + '.' + board[i+1:]
        board = board[:j] + new_p + board[j+1:]
        
        # Castling rights
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        
        # Castling
        if p == "K":
            wc = (False, False)
            if abs(j - i) == 2:
                kp = (i + j) // 2
                # Move rook
                rook_pos = A1 if j < i else H1
                rook_target = kp
                # Remove rook from original position
                wf -= pst['R'][rook_pos]
                bf -= pst['r'][119 - rook_pos]
                # Add rook to new position
                wf += pst['R'][rook_target]
                bf += pst['r'][119 - rook_target]
                # Update board
                board = board[:rook_pos] + '.' + board[rook_pos+1:]
                board = board[:rook_target] + 'R' + board[rook_target+1:]
        
        # Pawn promotion, double move and en passant capture
        if p == "P":
            if A8 <= j <= H8:
                # Promotion handled above with new_p
                pass
            if j - i == 2 * N:
                ep = i + N
            if j == self.ep:
                # Remove captured pawn
                captured_pos = j + S
                captured_piece = board[captured_pos]
                wf -= pst[captured_piece][captured_pos]
                bf -= pst[captured_piece.swapcase()][119 - captured_pos]
                
                material -= PIECE_VALUES[captured_piece]
                board = board[:captured_pos] + '.' + board[captured_pos+1:]
        
        # neural network evaluation
        nn_score = compute_value(wf, bf)

        total_score = nn_score + material
        
        # Create new position
        new_pos = Position(board, wf, bf, total_score, wc, bc, ep, kp, material)
        return new_pos.rotate()
    
    # Add hash key
    @property
    def key(self):
        """Return a hashable representation of the position"""
        return (self.board, self.wc, self.bc, self.ep, self.kp)


###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper
Entry = namedtuple("Entry", "lower upper")


class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.nodes = 0
        self.stop_flag = False  

    def bound(self, pos, gamma, depth, ply, can_null=True):
        """ Let s* be the "true" score of the sub-tree we are searching.
            The method returns r, where
            if gamma >  s* then s* <= r < gamma  (A better upper bound)
            if gamma <= s* then gamma <= r <= s* (A better lower bound) 
            ply: 从根节点开始的深度（步数）"""
        
        if self.stop_flag:  
            return 0
        
        if depth < 0:
            return pos.score
        self.nodes += 1

        # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
        # calmness, and from this point on there is no difference in behaviour depending on
        # depth, so so there is no reason to keep different depths in the transposition table.
        depth = max(depth, 0)

        # Sunfish is a king-capture engine, so we should always check if we
        # still have a king. Notice since this is the only termination check,
        # the remaining code has to be comfortable with being mated, stalemated
        # or able to capture the opponent king.
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER + ply  

        # Look in the table if we have already searched this position before.
        # We also need to be sure, that the stored search was over the same
        # nodes as the current search.
        entry = self.tp_score.get((pos.key, depth, can_null), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= gamma: return entry.lower
        if entry.upper < gamma: return entry.upper

        # Let's not repeat positions. We don't chat
        # - at the root (can_null=False) since it is in history, but not a draw.
        # - at depth=0, since it would be expensive and break "futulity pruning".
        if can_null and depth > 0 and pos.key in self.history:
            return 0

        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        def moves():
            # First try not moving at all. We only do this if there is at least one major
            # piece left on the board, since otherwise zugzwangs are too dangerous.
            # FIXME: We also can't null move if we can capture the opponent king.
            # Since if we do, we won't spot illegal moves that could lead to stalemate.
            # For now we just solve this by not using null-move in very unbalanced positions.
            # TODO: We could actually use null-move in Quiescent_Search as well. Not sure it would be very useful.
            # But still.... We just have to move stand-pat to be before null-move.
            #if depth > 2 and can_null and any(c in pos.board for c in "RBNQ"):
            #if depth > 2 and can_null and any(c in pos.board for c in "RBNQ") and abs(pos.score) < 500:
            
            if self.stop_flag:  
                return
        

            if depth > 2 and can_null and abs(pos.score) < 500:
                yield None, -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3, ply+1)

            # For QSearch we have a different kind of null-move, namely we can just stop
            # and not capture anything else.
            if depth == 0:
                yield None, pos.score

            # Look for the strongest ove from last time, the hash-move.
            killer = self.tp_move.get(pos.key)

            # If there isn't one, try to find one with a more shallow search.
            # This is known as Internal Iterative Deepening (IID). We set
            # can_null=True, since we want to make sure we actually find a move.
            if not killer and depth > 2:
                self.bound(pos, gamma, depth - 3, ply, can_null=False)
                killer = self.tp_move.get(pos.key)

            # If depth == 0 we only try moves with high intrinsic score (captures and
            # promotions). Otherwise we do all moves. This is called quiescent search.
            
            qs_val = options["Quiescent_Search"]["value"]
            qs_a_val = options["Quiescent_Search_A"]["value"]
            val_lower = qs_val - depth * qs_a_val

            # Only play the move if it would be included at the current val-limit,
            # since otherwise we'd get search instability.
            # We will search it again in the main loop below, but the tp will fix
            # things for us.
            if killer:  # Consider killer moves even if below val_lower
                yield killer, -self.bound(pos.move(killer), 1 - gamma, depth - 1, ply+1)

            # Then all the other moves
            for move in pos.gen_moves():
                # Skip killer move as we already considered it
                if move == killer:
                    continue
                    
                yield move, -self.bound(pos.move(move), 1 - gamma, depth - 1, ply+1)

        # Run through the moves, shortcutting when possible
        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                # Save the move for pv construction and killer heuristic
                if move is not None:
                    self.tp_move[pos.key] = move
                break

        # Stalemate checking is a bit tricky: Say we failed low, because
        # we can't (legally) move and so the (real) score is -infty.
        # At the next depth we are allowed to just return r, -infty <= r < gamma,
        # which is normally fine.
        # However, what if gamma = -10 and we don't have any legal moves?
        # Then the score is actaully a draw and we should fail high!
        # Thus, if best < gamma and best < 0 we need to double check what we are doing.

        # We will fix this problem another way: We add the requirement to bound, that
        # it always returns MATE_UPPER if the king is capturable. Even if another move
        # was also sufficient to go above gamma. If we see this value we know we are either
        # mate, or stalemate. It then suffices to check whether we're in check.

        # Note that at low depths, this may not actually be true, since maybe we just pruned
        # all the legal moves. So sunfish may report "mate", but then after more search
        # realize it's not a mate after all. That's fair.

        # This is too expensive to test at depth == 0
        if depth > 2 and best == -MATE_UPPER:
            flipped = pos.rotate(nullmove=True)
            # Hopefully this is already in the TT because of null-move
            in_check = self.bound(flipped, MATE_UPPER, 0, ply) == MATE_UPPER
            best = -MATE_LOWER if in_check else 0

        # Table part 2
        if best >= gamma:
            self.tp_score[(pos.key, depth, can_null)] = Entry(best, entry.upper)
        if best < gamma:
            self.tp_score[(pos.key, depth, can_null)] = Entry(entry.lower, best)

        return best

    def search(self, history):
        """Iterative deepening MTD-bi search"""
        self.nodes = 0
        self.stop_flag = False  
        self.history = set(pos.key for pos in history[-100:])
        self.tp_score.clear()

        gamma = 0
        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply. We also can't start at 0, since
        # that's quiscent search, and we don't always play legal moves there.
        mate_found = False  
        for depth in range(1, 1000):
           
            if self.stop_flag:
                break
                
            
            if has_input():
                line = sys.stdin.readline().strip()
                if line == "stop":
                    self.stop_flag = True
                    break
                elif line == "quit":
                    self.stop_flag = True
                    global quit_flag
                    quit_flag = True
                    break
                    
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # 'while lower != upper' would work, but it's too much effort to spend
            # on what's probably not going to change the move played.
            lower, upper = -MATE_LOWER, MATE_LOWER
           
            eval_roughness = options["EVAL_ROUGHNESS"]["value"]
            found_mate = False 
            while lower < upper - eval_roughness:
                score = self.bound(history[-1], gamma, depth, 0, can_null=False)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
                
                yield depth, gamma, score, self.tp_move.get(history[-1].key)
                
                
                if score >= MATE_LOWER - 100 or score <= -MATE_LOWER + 100:
                    found_mate = True
                    break  
                    
                gamma = (lower + upper + 1) // 2

            
            if found_mate:
                mate_found = True
                break

        
        if mate_found:
           
            best_move = self.tp_move.get(history[-1].key)
            if best_move:
                
                score = self.bound(history[-1], gamma, depth, 0, can_null=False)
                yield depth, gamma, score, best_move


###############################################################################
# UCI User interface
###############################################################################

def parse(c):
    fil, rank = ord(c[0]) - ord("a"), int(c[1]) - 1
    return A1 + fil - 10 * rank

def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord("a")) + str(-rank + 1)

def parse_fen(fen_parts):
    """Parse FEN string into Position object"""
    board_str, color_char, castling_str, enpas_str = fen_parts[:4]
    
    # Convert FEN board to our 120-char format
    board = ""
    rows = board_str.split('/')
    for row in rows:
        for char in row:
            if char in '12345678':
                board += '.' * int(char)
            else:
                board += char
        board += '\n'
    
    # Create 120-char board with borders
    new_board = "         \n" * 2
    for i in range(8):
        row = board[i*9:(i+1)*9].strip()
        new_board += ' ' + row.ljust(8) + ' \n'
    new_board += "         \n" * 2
    
    # Compute features
    wf, bf = features(new_board)
    
    # Castling rights
    wc = ('Q' in castling_str, 'K' in castling_str)
    bc = ('k' in castling_str, 'q' in castling_str)
    
    # En passant
    ep = 0
    if enpas_str != '-':
        file_char, rank_char = enpas_str[0], enpas_str[1]
        file = ord(file_char) - ord('a')
        rank = int(rank_char)
        ep = 21 + file + (8 - rank) * 10
    
    kp = 0
    
    
    material = material_value(new_board)
    
    # Create position
    pos = Position(
        board=new_board,
        wf=wf,
        bf=bf,
        score=compute_value(wf, bf) + material,
        wc=wc,
        bc=bc,
        ep=ep,
        kp=kp,
        material=material
    )
    
    # Rotate if black to move
    if color_char == 'b':
        pos = pos.rotate()
    
    return pos


wf0, bf0 = features(initial)
initial_material = material_value(initial)
initial_pos = Position(
    board=initial,
    wf=wf0,
    bf=bf0,
    score=compute_value(wf0, bf0) + initial_material,
    wc=(True, True),
    bc=(True, True),
    ep=0,
    kp=0,
    material=initial_material
)
move_history = []   
current_pos = initial_pos  
quit_flag = False   


searcher = Searcher()

# main loop
while not quit_flag:
    
    line = sys.stdin.readline().strip()
    if not line:
        continue
    
    args = line.split()
    if not args:
        continue

    if args[0] == "uci":
        print("id name", version)
        print("id author", author)
        
        for name, opt in options.items():
            print(f"option name {name} type {opt['type']} "
                  f"default {opt['value']} min {opt['min']} max {opt['max']}")
        print("uciok")
        sys.stdout.flush()  
    elif args[0] == "isready":
        print("readyok")
        sys.stdout.flush()  

    elif args[0] == "setoption":
        
        if len(args) >= 4 and args[1] == "name" and args[3] == "value":
            opt_name = args[2]
            opt_value = args[4]
            if opt_name in options:
                try:
                    value = int(opt_value)
                    if options[opt_name]["min"] <= value <= options[opt_name]["max"]:
                        options[opt_name]["value"] = value
                    else:
                        print(f"info string Invalid value for {opt_name}: {value} out of range")
                except ValueError:
                    print(f"info string Invalid value for {opt_name}: {opt_value} is not an integer")
            else:
                print(f"info string Unknown option: {opt_name}")
        sys.stdout.flush()  
    elif args[0] == "quit":
        quit_flag = True
        break

    elif args[:2] == ["position", "startpos"]:
        
        move_history = args[3:] if len(args) > 3 else []
        
        current_pos = initial_pos

    elif args[0] == "position" and args[1] == "fen":
        
        fen_parts = args[2:8]
        try:
            current_pos = parse_fen(fen_parts)
            
            if len(args) > 8 and args[8] == "moves":
                move_history = args[9:]
            else:
                move_history = []
        except Exception as e:
            print(f"info string Error parsing FEN: {e}")
            continue

    elif args[0] == "stop":
        
        searcher.stop_flag = True

    elif args[0] == "go":
        
        hist = [current_pos]
        for ply, move_str in enumerate(move_history):
            i = parse(move_str[:2])
            j = parse(move_str[2:4])
            prom = move_str[4:].upper() if len(move_str) > 4 else ""
            
            if ply % 2 == 1:
                i, j = 119 - i, 119 - j
            hist.append(hist[-1].move(Move(i, j, prom)))
        
        
        max_movetime = 0
        max_depth = 1000
        max_nodes = 0
        
        
        i = 1
        while i < len(args):
            if args[i] == "movetime":
                max_movetime = int(args[i+1]) / 1000  
                i += 2
            elif args[i] == "depth":
                max_depth = int(args[i+1])
                i += 2
            elif args[i] == "nodes":
                max_nodes = int(args[i+1])
                i += 2
            elif args[i] == "infinite":
                max_movetime = 10**6
                i += 1
            else:
                i += 1
        
        start = time.time()
        move_str = None
        
       
        for depth, gamma, score, move in searcher.search(hist):
            
            elapsed = time.time() - start
            if max_movetime and elapsed >= max_movetime:
                break
            if max_depth and depth >= max_depth:
                break
            if max_nodes and searcher.nodes >= max_nodes:
                break
            if searcher.stop_flag:  
                break
            
        
            if score >= MATE_LOWER:  
                moves_to_mate = MATE - score 
                score_str = f"mate {moves_to_mate}"
            elif score <= -MATE_LOWER:  
                moves_to_mate = MATE + score
                score_str = f"mate -{moves_to_mate}"
            else:
                score_str = f"cp {score}"
            
            
            if move is not None:
                i, j = move.i, move.j
                if len(hist) % 2 == 0:  
                    i, j = 119 - i, 119 - j
                move_str = render(i) + render(j) + move.prom.lower()
                print("info depth", depth, "score", score_str, "nodes", searcher.nodes, 
                      "time", int(elapsed*1000), "pv", move_str)
                sys.stdout.flush()  
            else:
                print("info depth", depth, "score", score_str, "nodes", searcher.nodes, 
                      "time", int(elapsed*1000))
                sys.stdout.flush()  
       
        if move_str:
            print("bestmove", move_str)
        else:
           
            for move in hist[-1].gen_moves():
                i, j = move.i, move.j
                if len(hist) % 2 == 0:
                    i, j = 119 - i, 119 - j
                move_str = render(i) + render(j) + move.prom.lower()
                print("bestmove", move_str)
                break
            else:
                print("bestmove (none)")
        sys.stdout.flush()  