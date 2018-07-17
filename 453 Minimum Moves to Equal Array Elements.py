class Answer(object):
'''453. Minimum Moves to Equal Array Elements'''
    def minMoves(nums): return sum(nums) - len(nums) * min(nums)