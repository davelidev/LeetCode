class Answer(object):
'''416. Partition Equal Subset Sum'''
    def canPartition(nums):
        all_sums = {0}
        for num in nums: all_sums |= set(part_total + num for part_total in all_sums)
        return (float(sum(nums)) / 2) in all_sums