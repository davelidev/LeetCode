class Answer(object):
'''848. Shifting Letters'''
    def shiftingLetters(S, shifts):
        cur = 0
        for i in range(len(shifts) -1, -1, -1):
            shifts[i] += cur
            cur = shifts[i]
        return ''.join(chr((ord(S[i]) - ord('a') + shifts[i]) % 26 + ord('a'))
                       for i in range(len(shifts)))