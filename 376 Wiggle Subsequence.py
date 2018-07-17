class Answer(object):
'''376. Wiggle Subsequence'''
    def wiggleMaxLength(nums):
        # update up_tog when the first up/down, or down/up is seen.
        # if up_tog is set, and at the current position, it goes in the
        # same direction of up_tog, then we toggle the direction and 
        # add to the count.
        if not nums: return 0
        count = 1
        up_tog = None
        for i in range(1, len(nums)):
            if up_tog is None and nums[i-1] != nums[i]:
                up_tog = nums[i-1] < nums[i]
            if (up_tog is not None) and              (nums[i-1] < nums[i] and up_tog or               nums[i-1] > nums[i] and not up_tog):
                count, up_tog = count + 1, not up_tog
        return count