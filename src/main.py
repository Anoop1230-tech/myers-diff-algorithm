"""Minimal line diff (Part A) and changed-character highlighting (Part B).

Usage:
    main.py lines     A B
    main.py highlight A B

The diff is Myers' O(ND) algorithm in its linear-space form (section 4b of
the paper): find a point in the middle of an optimal edit path by searching
forwards from the start and backwards from the end at the same time, split
the problem there, and repeat on both halves. The same code diffs lines
(Part A) and the characters of a changed line pair (Part B).
"""

import sys
from itertools import compress


# ---------------------------------------------------------------------------
# Reading input
# ---------------------------------------------------------------------------

def read_lines(path):
    """Read a file as raw bytes and split it into lines on b"\\n".

    A final newline does not create an extra empty line, an empty file has
    no lines, and any b"\\r" stays part of its line.
    """
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    if lines[-1] == b"":  # split() always returns at least one piece
        lines.pop()
    return lines


# ---------------------------------------------------------------------------
# Snakes: runs of equal elements along a diagonal
# ---------------------------------------------------------------------------

def snake_forward(a, x, b, y, x_end, y_end):
    """Follow equal elements a[x] == b[y], a[x+1] == b[y+1], ... forwards.

    Returns the x where the run stops (y moves by the same amount). It
    compares slices whose length doubles while they match, so a long run
    costs a few C-level comparisons instead of one Python step per element.
    """
    step = 1
    while True:
        lim = min(x_end - x, y_end - y, step)
        if lim <= 0:
            return x
        if a[x:x + lim] == b[y:y + lim]:
            x += lim
            y += lim
            step += step
        elif lim == 1:
            return x
        else:
            step = lim >> 1


def snake_backward(a, x, b, y, x_start, y_start):
    """Count equal elements going backwards: a[x-1] == b[y-1], a[x-2] == b[y-2], ...

    Never goes below x_start / y_start. Returns the length of the run.
    """
    step = 1
    count = 0
    while True:
        lim = min(x - count - x_start, y - count - y_start, step)
        if lim <= 0:
            return count
        hi_a = x - count
        hi_b = y - count
        if a[hi_a - lim:hi_a] == b[hi_b - lim:hi_b]:
            count += lim
            step += step
        elif lim == 1:
            return count
        else:
            step = lim >> 1


# ---------------------------------------------------------------------------
# Myers: find a point on an optimal edit path (the "middle snake")
# ---------------------------------------------------------------------------

def middle_split(a, b):
    """Return a point (x, y) in the middle of a minimal edit path from (0, 0)
    to (n, m), or None if a and b have nothing in common.

    a and b must be non-empty, and their first and last elements must differ
    (the caller has trimmed the common prefix and suffix), so the returned
    point always splits the problem into two strictly smaller ones.

    Diagonal k holds the points with x - y == k. vf[k] is the furthest x
    reached on diagonal k by a forward path with d edits. vb[k] is the same
    for the backward search, which runs the same steps on the reversed
    sequences, so its x and y count elements from the end. Forward diagonal
    k and backward diagonal delta - k are the same line of the edit graph.
    Negative k uses Python's negative list indices, so V needs no offset.
    """
    n = len(a)
    m = len(b)
    ar = a[::-1]
    br = b[::-1]
    max_d = (n + m + 1) // 2
    size = 2 * max_d + 4        # room for diagonals -(max_d+1) .. max_d+1
    vf = [-1] * size            # -1 means "not reached yet"
    vb = [-1] * size
    vf[1] = 0                   # lets d = 0 start at (0, 0) through the rule
    vb[1] = 0
    delta = n - m
    odd = delta & 1             # odd delta: the forward pass finds the overlap
    # Diagonals that ran off the edit graph are dropped from later rounds.
    f_lo = f_hi = b_lo = b_hi = 0

    for d in range(max_d + 1):
        # ---- forward search, d edits ----
        for k in range(-d + f_lo, d + 1 - f_hi, 2):
            # Take the neighbour diagonal that got further: down from k + 1
            # (an insertion) keeps x, right from k - 1 (a deletion) adds 1.
            # This is the paper's rule; unreached neighbours hold -1, so at
            # k == -d and k == d the missing neighbour is never chosen.
            x = vf[k + 1]
            t = vf[k - 1] + 1
            if t > x:
                x = t
            y = x - k
            if x < n and y < m and a[x] == b[y]:
                x = snake_forward(a, x + 1, b, y + 1, n, m)
                y = x - k
            vf[k] = x
            if x > n:
                f_hi += 2           # ran off the right edge
            elif y > m:
                f_lo += 2           # ran off the bottom edge
            elif odd:
                c = delta - k       # same line, backward numbering
                if -d < c < d and vb[c] != -1 and x + vb[c] >= n:
                    return x, y     # the forward path met a backward path
        # ---- backward search, d edits (on the reversed sequences) ----
        for k in range(-d + b_lo, d + 1 - b_hi, 2):
            x = vb[k + 1]
            t = vb[k - 1] + 1
            if t > x:
                x = t
            y = x - k
            if x < n and y < m and ar[x] == br[y]:
                x = snake_forward(ar, x + 1, br, y + 1, n, m)
                y = x - k
            vb[k] = x
            if x > n:
                b_hi += 2
            elif y > m:
                b_lo += 2
            elif not odd:
                c = delta - k       # same line, forward numbering
                if -d <= c <= d:
                    fx = vf[c]
                    if fx != -1 and fx + x >= n:
                        return fx, fx - c
    return None


# ---------------------------------------------------------------------------
# Longest common subsequence by divide and conquer
# ---------------------------------------------------------------------------

def common_runs(a, b):
    """Runs (i, j, length) with a[i:i+length] == b[j:j+length] that together
    form a longest common subsequence of a and b, sorted by i.

    Fewest deletions + insertions = n + m - 2 * (total length of the runs).
    An explicit stack replaces recursion so deep splits cannot overflow.
    """
    runs = []
    stack = [(0, len(a), 0, len(b))]
    while stack:
        a_lo, a_hi, b_lo, b_hi = stack.pop()
        # Trim the common prefix.
        if a_lo < a_hi and b_lo < b_hi and a[a_lo] == b[b_lo]:
            end = snake_forward(a, a_lo, b, b_lo, a_hi, b_hi)
            runs.append((a_lo, b_lo, end - a_lo))
            b_lo += end - a_lo
            a_lo = end
        # Trim the common suffix.
        if a_lo < a_hi and b_lo < b_hi and a[a_hi - 1] == b[b_hi - 1]:
            run = snake_backward(a, a_hi, b, b_hi, a_lo, b_lo)
            a_hi -= run
            b_hi -= run
            runs.append((a_hi, b_hi, run))
        if a_lo == a_hi or b_lo == b_hi:
            continue                # only deletions or only insertions remain
        split = middle_split(a[a_lo:a_hi], b[b_lo:b_hi])
        if split is None:
            continue                # nothing in common
        x, y = split
        stack.append((a_lo + x, a_hi, b_lo + y, b_hi))
        stack.append((a_lo, a_lo + x, b_lo, b_lo + y))
    runs.sort()
    return runs


def diff_runs(a, b):
    """Kept runs (i, j, length) of a minimal diff of two sequences (lists of
    lines, or strings of characters), sorted, with touching runs merged.
    Everything outside the runs is deleted from a or inserted from b.

    Before running Myers it drops items that never occur in the other
    sequence: they can never be kept, so a longest common subsequence of
    what is left is also one of the full sequences, and the search is
    smaller. The surviving positions are mapped back at the end.
    """
    n = len(a)
    m = len(b)
    pre = 0
    if n and m and a[0] == b[0]:
        pre = snake_forward(a, 0, b, 0, n, m)
    suf = 0
    if pre < n and pre < m and a[n - 1] == b[m - 1]:
        suf = snake_backward(a, n, b, m, pre, pre)

    runs = []
    if pre:
        runs.append((0, 0, pre))
    mid_a = a[pre:n - suf]
    mid_b = b[pre:m - suf]
    if mid_a and mid_b:
        common = set(mid_a).intersection(mid_b)
        if common:
            in_a = list(map(common.__contains__, mid_a))
            in_b = list(map(common.__contains__, mid_b))
            pos_a = list(compress(range(pre, n - suf), in_a))  # original index
            pos_b = list(compress(range(pre, m - suf), in_b))
            for i, j, length in common_runs(list(compress(mid_a, in_a)),
                                            list(compress(mid_b, in_b))):
                add_mapped_run(pos_a, pos_b, i, j, length, runs)
    if suf:
        runs.append((n - suf, m - suf, suf))

    merged = []
    for i, j, length in runs:
        if merged:
            pi, pj, pl = merged[-1]
            if pi + pl == i and pj + pl == j:
                merged[-1] = (pi, pj, pl + length)
                continue
        merged.append((i, j, length))
    return merged


def add_mapped_run(pos_a, pos_b, i, j, length, runs):
    """Map a run of the filtered sequences back to original positions.

    A run stays one run where the original positions are consecutive on
    both sides; otherwise it is halved until each piece is consecutive.
    """
    todo = [(i, j, length)]
    while todo:
        i, j, length = todo.pop()
        last = length - 1
        if pos_a[i + last] - pos_a[i] == last and pos_b[j + last] - pos_b[j] == last:
            runs.append((pos_a[i], pos_b[j], length))
        else:
            half = length >> 1
            todo.append((i + half, j + half, length - half))
            todo.append((i, j, half))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def ranges_text(runs, side, length):
    """Changed positions of one side as merged ranges "s-e,s-e", or ".".

    side is 0 for the old line and 1 for the new one. The gaps between the
    kept runs are exactly the changed characters; two gaps never touch,
    because a kept character always lies between them.
    """
    parts = []
    pos = 0
    for run in runs:
        start = run[side]
        if start > pos:
            parts.append("%d-%d" % (pos, start))
        pos = start + run[2]
    if length > pos:
        parts.append("%d-%d" % (pos, length))
    return ",".join(parts) if parts else "."


def highlight_line(old, new):
    """The "? old | new" line for one paired - / + line, as bytes."""
    old_s = old.decode("utf-8", "surrogateescape")   # code points
    new_s = new.decode("utf-8", "surrogateescape")
    runs = diff_runs(old_s, new_s)
    text = "? %s | %s\n" % (ranges_text(runs, 0, len(old_s)),
                            ranges_text(runs, 1, len(new_s)))
    return text.encode("ascii")


def render(a, b, highlight):
    """Build the whole output as bytes.

    Between two kept runs is one change block: its "-" lines are printed
    before its "+" lines. In highlight mode the t-th "+" line of a block is
    paired with the t-th "-" line and followed by its "?" line.
    """
    out = []
    add = out.append
    prev_i = prev_j = 0
    for i, j, length in diff_runs(a, b) + [(len(a), len(b), 0)]:
        if i > prev_i:
            add(b"-" + b"\n-".join(a[prev_i:i]) + b"\n")
        if j > prev_j:
            if highlight and i > prev_i:
                for t in range(j - prev_j):
                    add(b"+" + b[prev_j + t] + b"\n")
                    if prev_i + t < i:
                        add(highlight_line(a[prev_i + t], b[prev_j + t]))
            else:
                add(b"+" + b"\n+".join(b[prev_j:j]) + b"\n")
        if length:
            add(b" " + b"\n ".join(a[i:i + length]) + b"\n")
        prev_i = i + length
        prev_j = j + length
    return b"".join(out)


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("lines", "highlight"):
        print("usage: main.py lines|highlight A_PATH B_PATH", file=sys.stderr)
        return 2
    command, a_path, b_path = sys.argv[1:]
    try:
        a = read_lines(a_path)
        b = read_lines(b_path)
    except OSError as err:
        print("error: cannot read input: %s" % err, file=sys.stderr)
        return 2
    sys.stdout.buffer.write(render(a, b, command == "highlight"))
    sys.stdout.buffer.flush()
    return 0


raise SystemExit(main())
