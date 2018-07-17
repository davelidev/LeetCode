class Answer(object):
'''260. Single Number III'''
    def singleNumber(nums):
        xord = 0
        for num in nums: xord ^= num

        for i in range(32):
            if (1 << i) & xord:
                num1 = 0
                num2 = 0
                for num in nums:
                    if (1 << i) & num:
                        num1 ^= num
                    else:
                        num2 ^= num
                return [num1, num2]