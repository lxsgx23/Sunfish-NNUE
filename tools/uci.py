# Advanced UCI interface

import re, time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from functools import partial

print = partial(print, flush=True)


def render_move(move, white_pov):
    if move is None:
        return "(none)"
    i, j = move.i, move.j
    if not white_pov:
        i, j = 119 - i, 119 - j
    render = sunfish.render
    return render(i) + render(j) + move.prom.lower()


def parse_move(move_str, white_pov):
    parse = sunfish.parse
    i, j, prom = parse(move_str[:2]), parse(move_str[2:4]), move_str[4:].upper()
    if not white_pov:
        i, j = 119 - i, 119 - j
    return sunfish.Move(i, j, prom)


def go_loop(searcher, hist, stop_event, max_movetime=0, max_depth=0, max_nodes=0, debug=False):
    if debug:
        print(f"Going movetime={max_movetime}, depth={max_depth}, nodes={max_nodes}")

    start = time.time()
    best_move = None
    for depth, gamma, score, move in searcher.search(hist):
        # 检查停止条件
        elapsed = time.time() - start
        if max_movetime and elapsed > max_movetime:
            break
        if max_depth and depth >= max_depth:
            break
        if max_nodes and searcher.nodes >= max_nodes:
            break
        if stop_event.is_set():
            break
            
        # 输出搜索信息
        fields = {
            "depth": depth,
            "time": round(1000 * elapsed),
            "nodes": searcher.nodes,
            "nps": round(searcher.nodes / (elapsed+1)),
        }
        if score >= gamma:
            fields["score cp"] = f"{score} lowerbound"
            best_move = render_move(move, white_pov=len(hist) % 2 == 1)
            fields["pv"] = " ".join(pv(searcher, hist[-1], include_scores=False))
        else:
            fields["score cp"] = f"{score} upperbound"
        print("info", " ".join(f"{k} {v}" for k, v in fields.items()))

    # 输出最佳移动
    my_pv = pv(searcher, hist[-1], include_scores=False)
    print("bestmove", my_pv[0] if my_pv else "(none)")


def mate_loop(
    searcher,
    hist,
    stop_event,
    max_movetime=0,
    max_depth=0,
    find_draw=False,
    debug=False,
):
    start = time.time()
    for d in range(int(max_depth) + 1):
        if find_draw:
            s0 = searcher.bound(hist[-1], 0, d)
            elapsed = time.time() - start
            print("info", "depth", d, "score lowerbound cp", s0)
            s1 = searcher.bound(hist[-1], 1, d)
            elapsed = time.time() - start
            print("info", "depth", d, "score upperbound cp", s1)
            if s0 >= 0 and s1 < 1:
                break
        else:
            score = searcher.bound(hist[-1], sunfish.MATE_LOWER, d)
            elapsed = time.time() - start
            print(
                "info depth",
                d,
                "score cp",
                score,
                "time",
                round(1000 * elapsed),
                "pv",
                " ".join(pv(searcher, hist[-1], include_scores=False)),
            )
            if score >= sunfish.MATE_LOWER:
                break
        if elapsed > max_movetime:
            break
        if stop_event.is_set():
            break
    move = searcher.tp_move.get(hist[-1])
    move_str = render_move(move, white_pov=len(hist) % 2 == 1)
    print("bestmove", move_str)


def perft(pos, depth, debug=False):

    def _perft_count(pos, depth):
        # Check that we didn't get to an illegal position
        if can_kill_king(pos):
            return -1
        if depth == 0:
            return 1
        res = 0
        for move in pos.gen_moves():
            cnt = _perft_count(pos.move(move), depth - 1)
            if cnt != -1:
                res += cnt
        return res

    total = 0
    for move in pos.gen_moves():
        move_uci = render_move(move, get_color(pos) == WHITE)
        cnt = _perft_count(pos.move(move), depth - 1)
        if cnt != -1:
            print(f"{move_uci}: {cnt}")
            total += cnt
    print()
    print("Nodes searched:", total)


def run(sunfish_module, startpos):
    global sunfish
    sunfish = sunfish_module

    debug = False
    hist = [startpos]
    searcher = sunfish.Searcher()

    with ThreadPoolExecutor(max_workers=1) as executor:
        # Noop future to get started
        go_future = executor.submit(lambda: None)
        do_stop_event = Event()

        while True:
            try:
                args = input().split()
                if not args:
                    continue

                elif args[0] in ("stop", "quit"):
                    if go_future.running():
                        if debug:
                            print("Stopping go loop...")
                        do_stop_event.set()
                        go_future.result()
                    else:
                        if debug:
                            print("Go loop not running...")
                    if args[0] == "quit":
                        break

                elif not go_future.done():
                    print(f"Ignoring input {args}. Please call 'stop' first.")
                    continue

                # Make sure we are really done, and throw any errors that may have
                # happened in the go loop.
                go_future.result(timeout=0)

                if args[0] == "uci":
                    print(f"id name {sunfish.version}")
                    print(f"id author {sunfish.author}")
                    for attr, (lo, hi) in sunfish.opt_ranges.items():
                        default = getattr(sunfish, attr)
                        print(
                            f"option name {attr} type spin default {default} min {lo} max {hi}"
                        )
                    print("uciok")

                elif args[0] == "setoption":
                    _, uci_key, _, uci_value = args[1:]
                    setattr(sunfish, uci_key, int(uci_value))

                # FIXME: It seems we should reply to "isready" even while thinking.
                # See: https://talkchess.com/forum3/viewtopic.php?f=7&t=81233&start=10
                elif args[0] == "isready":
                    print("readyok")

                elif args[0] == "quit":
                    break

                elif args[:2] == ["position", "startpos"]:
                    hist = [startpos]
                    for ply, move in enumerate(args[3:]):
                        hist.append(hist[-1].move(parse_move(move, ply % 2 == 0)))

                elif args[:2] == ["position", "fen"]:
                    pos = from_fen(*args[2:8])
                    hist = [pos] if get_color(pos) == WHITE else [pos.rotate(), pos]
                    if len(args) > 8:
                        assert args[8] == "moves"
                        for move in args[9:]:
                            hist.append(hist[-1].move(parse_move(move, len(hist) % 2 == 1)))

                elif args[0] == "go":
                    think = 10**6
                    max_depth = 1000
                    max_nodes = 0
                    loop = go_loop

                    # 解析go命令参数
                    i = 1
                    while i < len(args):
                        if args[i] == "movetime":
                            think = int(args[i+1]) / 1000
                            i += 2
                        elif args[i] == "depth":
                            max_depth = int(args[i+1])
                            i += 2
                        elif args[i] == "nodes":
                            max_nodes = int(args[i+1])
                            i += 2
                        elif args[i] == "infinite":
                            think = 10**6
                            i += 1
                        elif args[i] == "wtime":
                            wtime, btime, winc, binc = [int(a) / 1000 for a in args[i+1:i+5]]
                            # 根据当前玩家设置时间
                            if len(hist) % 2 == 0:  # 黑方回合
                                wtime, winc = btime, binc
                            think = min(wtime / 40 + winc, wtime / 2 - 1)
                            # 开局快速移动
                            if len(hist) < 3:
                                think = min(think, 1)
                            i += 5
                        else:
                            i += 1

                    do_stop_event.clear()
                    go_future = executor.submit(
                        loop,
                        searcher,
                        hist,
                        do_stop_event,
                        think,
                        max_depth,
                        max_nodes,
                        debug=debug,
                    )


                    # Make sure we get informed if the job fails
                    def callback(fut):
                        fut.result(timeout=0)

                    go_future.add_done_callback(callback)

            except (KeyboardInterrupt, EOFError):
                if go_future.running():
                    if debug:
                        print("Stopping go loop...")
                    do_stop_event.set()
                    go_future.result()
                break


# Old tools stuff

WHITE, BLACK = range(2)


def from_fen(board, color, castling, enpas, _hclock, _fclock):
    board = re.sub(r"\d", (lambda m: "." * int(m.group(0))), board)
    board = list(21 * " " + "  ".join(board.split("/")) + 21 * " ")
    board[9::10] = ["\n"] * 12
    board = "".join(board)
    wc = ("Q" in castling, "K" in castling)
    bc = ("k" in castling, "q" in castling)
    ep = sunfish.parse(enpas) if enpas != "-" else 0
    if hasattr(sunfish, 'features'):
        wf, bf = sunfish.features(board)
        pos = sunfish.Position(board, 0, wf, bf, wc, bc, ep, 0)
        pos = pos._replace(score=pos.calculate_score())
    else:
        score = sum(sunfish.pst[c][i] for i, c in enumerate(board) if c.isupper())
        score -= sum(sunfish.pst[c.upper()][119-i] for i, c in enumerate(board) if c.islower())
        pos = sunfish.Position(board, score, wc, bc, ep, 0)
    return pos if color == 'w' else pos.rotate()


def get_color(pos):
    """A slightly hacky way to to get the color from a sunfish position"""
    return BLACK if pos.board.startswith("\n") else WHITE


def can_kill_king(pos):
    # If we just checked for opponent moves capturing the king, we would miss
    # captures in case of illegal castling.
    #MATE_LOWER = 60_000 - 10 * 929
    #return any(pos.value(m) >= MATE_LOWER for m in pos.gen_moves())
    return any(pos.board[m.j] == 'k' or abs(m.j - pos.kp) < 2 for m in pos.gen_moves())


def pv(searcher, pos, include_scores=True, include_loop=False):
    res = []
    seen_pos = set()
    color = get_color(pos)
    origc = color
    if include_scores:
        res.append(str(pos.score))
    while True:
        if hasattr(pos, "wf"):
            move = searcher.tp_move.get(pos.hash())
        elif hasattr(searcher, "tp_move"):
            move = searcher.tp_move.get(pos)
        elif hasattr(searcher, "tt_new"):
            move = searcher.tt_new[0][pos, True].move
        # The tp may have illegal moves, given lower depths don't detect king killing
        if move is None or can_kill_king(pos.move(move)):
            break
        res.append(render_move(move, get_color(pos) == WHITE))
        pos, color = pos.move(move), 1 - color

        if hasattr(pos, "wf"):
            if pos.hash() in seen_pos:
                if include_loop:
                    res.append("loop")
                break
            seen_pos.add(pos.hash())
        else:
            if pos in seen_pos:
                if include_loop:
                    res.append("loop")
                break
            seen_pos.add(pos)

        if include_scores:
            res.append(str(pos.score if color == origc else -pos.score))
    return res
