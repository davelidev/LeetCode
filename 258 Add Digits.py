class Answer(object):
'''258. Add Digits'''
    def addDigits(num):
        while (num / 10):
            n_num = 0
            while num:
                num, mod = divmod(num, 10)
                n_num += mod
            num = n_num
        return num