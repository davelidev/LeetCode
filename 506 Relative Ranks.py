class Answer(object):
'''506. Relative Ranks'''
    def findRelativeRanks(nums):
        sorted_scores = sorted(nums, reverse=True)
        str_scores = ["Gold Medal", "Silver Medal", "Bronze Medal"]
        str_scores = (str_scores + [str(i) for i in range(4, len(nums) + 1)])[:len(nums)]
        score_to_str = dict(zip(sorted_scores, str_scores))
        return [score_to_str[score] for score in nums]