class Answer(object):
'''389. Find the Difference'''
    def findTheDifference(s, t):
        return chr(reduce(operator.xor, [ord(char) for char in s + t], 0))