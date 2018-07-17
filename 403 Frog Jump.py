class Answer(object):
'''403. Frog Jump'''
    def canCross(stones):
        from collections import defaultdict
        stone_to_steps = defaultdict(set)
        if (stones[1] - stones[0]) != 1: return False
        stone_to_steps[stones[1]].add(1)
        for pos in stones:
            for step in stone_to_steps[pos]:
                for new_pos in [pos + step + i for i in [-1, 0, 1]]:
                    if new_pos == stones[-1]: return True
                    elif new_pos != pos:
                        stone_to_steps[new_pos].add(new_pos - pos)
        return False