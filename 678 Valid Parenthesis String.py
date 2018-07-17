class Answer(object):
'''678. Valid Parenthesis String'''
    def checkValidString(s):
        high = low = 0
        for i, char in enumerate(s):
            high += -1 if char == ')' else 1
            low = low + 1 if char == '(' else max(low - 1, 0)
            if high < 0:
                return False
        return low == 0