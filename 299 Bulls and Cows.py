class Answer(object):
'''299. Bulls and Cows'''
    def getHint(secret, guess):
        a = 0
        s_counts = {}
        g_counts = {}
        for s_c, g_c in zip(secret, guess):
            if s_c == g_c: a += 1
            else:
                s_counts[s_c] = s_counts.get(s_c, 0) + 1
                g_counts[g_c] = g_counts.get(g_c, 0) + 1
        b = sum(min(s_counts.get(g_c, 0), g_counts[g_c]) for g_c in g_counts)
        return "%dA%dB" %(a, b)