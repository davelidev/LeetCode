class Answer(object):
'''367. Valid Perfect Square'''
    def isPerfectSquare(num):
        diff = 3
        sq = 1
        while sq < num:
            sq += diff
            diff += 2
        return sq == num