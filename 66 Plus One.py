class Answer(object):'''66. Plus One'''
    def plusOne(digits):
        for i in range(len(digits) - 1, -1, -1):
            if digits[i] < 9:
                digits[i] += 1
                return digits
            else: digits[i] = 0
        digits.insert(0, 1)
        return digits