class Answer(object):
'''167. Two Sum II - Input array is sorted'''
    def twoSum(numbers, target):
        i, j = 0, len(numbers) - 1
        while i < j:
            pair_sum = numbers[i] + numbers[j]
            if pair_sum == target: return i + 1, j + 1
            elif pair_sum < target: i += 1
            else: j -= 1