class Answer(object):
'''169. Majority Element'''
    def majorityElement(nums):
        count, most_freq_elem = 0, None
        for num in nums:
            if most_freq_elem is None:
                count, most_freq_elem = 1, num
            elif num != most_freq_elem:
                count -= 1
                if count == 0:
                    count, most_freq_elem = 1, num
            elif num == most_freq_elem:
                count += 1
        return most_freq_elem