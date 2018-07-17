class Answer(object):
'''228. Summary Ranges'''
    def summaryRanges(nums):
        ranges = []
        for i in range(len(nums)):
            if not ranges or ranges[-1][1] != nums[i] - 1:
                ranges.append([nums[i], nums[i]])
            else:
                ranges[-1][1] = nums[i]
        return [str(i) + '->' + str(j) if i != j else str(i) for i, j in ranges]