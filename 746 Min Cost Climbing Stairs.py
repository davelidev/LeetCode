class Answer(object):
'''746. Min Cost Climbing Stairs'''
    def minCostClimbingStairs(cost):
        if len(cost) <= 2: return int(bool(len(cost)))
        dp_cost_step = cost[:2]
        for i in range(2, len(cost)):
            dp_cost_step.append(min(dp_cost_step[-2:]) + cost[i])
        return min(dp_cost_step[-2:])