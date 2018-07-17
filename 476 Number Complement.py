class Answer(object):
'''476. Number Complement'''
    def findComplement(self, num):
        mask = 0
        while mask & num != num: mask = (mask << 1) | 1
        return num ^ mask