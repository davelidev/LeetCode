class Answer(object):
'''89. Gray Code'''
    def grayCode(n):
        gray = [0]
        # take the reverse so that the last of gray is the same as beginning of gray.
        # Loop n times to generate the integers of length n in binary
        for i in range(n):
            gray.extend([g | (1 << i) for g in reversed(gray)])
        return gray