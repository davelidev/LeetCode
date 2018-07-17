class Answer(object):
'''134. Gas Station'''
    def canCompleteCircuit(gas, cost):
        diff = [gas[i] - cost[i] for i in range(len(gas))]
        if len(diff) == 1:
            if diff[0] >= 0:
                return 0
            else:
                return -1
        start_idx = 0
        end_idx = 0
        if not diff:
            return True
        # keep acc positive
        acc = diff[start_idx]
        for i in range(len(diff) - 1):
            if acc <= 0:
                start_idx -= 1
                acc += diff[start_idx % len(diff)]
            else:
                end_idx += 1
                acc += diff[end_idx % len(diff)]
        if acc >= 0:
            return start_idx % len(diff)
        else:
            return -1