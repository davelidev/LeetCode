class Answer(object):'''55. Jump Game'''
    def canJump(nums):
        max_idx_jump = 0
        for i in range(len(nums)):
            if max_idx_jump < i:
                return False
            max_idx_jump = max(max_idx_jump, nums[i] + i)
        return True