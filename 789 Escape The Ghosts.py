class Answer(object):'''789. Escape The Ghosts'''
    def escapeGhosts(ghosts, target):
        def dist(x, y): return sum(map(abs, [target[0] - x, target[1] - y]))
        dist_ghost = dist(0, 0)
        return not any(dist(x, y) <= dist_ghost for x, y in ghosts)