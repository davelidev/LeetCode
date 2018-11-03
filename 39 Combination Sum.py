class Answer(object):'''39. Combination Sum'''
    def combinationSum(candidates, target):
        dp = [[0, []]]
        for num in candidates:
            dp = [ [cur_target + num * i, lst + ([num] * i)]
                        for cur_target, lst in dp for i in range((target - cur_target) / num + 1)]
        return [lst for num, lst in dp if num == target]