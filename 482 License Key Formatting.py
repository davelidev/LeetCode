class Answer(object):
'''482. License Key Formatting'''
    def licenseKeyFormatting(S, K):
        S = S.replace('-', '').upper()
        return '-'.join([ S[max(len(S) - i, 0) : len(S) - i + K]
                         for i in range(K, len(S) + K, K)][::-1])