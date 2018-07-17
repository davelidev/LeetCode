class Answer(object):
'''628. Maximum Product of Three Numbers'''
    def maximumProduct(nums):
        max1 = max2 = max3 = float('-inf')
        min1 = min2 = float('inf')
        for num in nums:
            
            if num > max1: max1, max2, max3 = num, max1, max2
            elif num > max2: max2, max3 = num, max2
            elif num > max3: max3 = num
                
            if num < min1: min1, min2 = num, min1
            elif num < min2: min2 = num
        return max(max1 * max2 * max3, min1 * min2 * max1)

    def maximumProduct(nums):
        max3, min2 = [], []
        for num in nums:
            if (not max3.append(num)) and len(max3) == 4: max3.remove(min(max3))
            if (not min2.append(num)) and len(min2) == 3: min2.remove(max(min2))
        def prod(lst): return reduce(lambda x, y: x * y, lst, 1)
        return max(prod(max3), prod(min2 + [max(max3)]))