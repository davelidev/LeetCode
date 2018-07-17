class Answer(object):
'''507. Perfect Number'''
    def checkPerfectNumber(num):
        if num <= 0: return False
        sqrt = num ** 0.5
        div_sum = sum(j for i in range(2, int(sqrt) + 1) if num % i == 0 for j in [i, num / i]) + 1
        if int(sqrt) == sqrt: div_sum -= sqrt
        return div_sum == num