class Answer(object):
'''477. Total Hamming Distance'''
    def totalHammingDistance(nums):
        nums = [[int(bool(num & (1 << i))) for i in range(31, -1, -1)] for num in nums]
        counts = [sum([num[i] for num in nums]) for i in range(31, -1, -1)]
        return sum([count * (len(nums) - count) for count in counts])