class Answer(object):
'''718. Maximum Length of Repeated Subarray'''
    def findLength(A, B):
        if not A or not B: return 0
        dp = [[0] * (len(B) + 1) for _ in range((len(A)) + 1)]
        for i in range(len(A)):
            for j in range(len(B)):
                if A[i] == B[j]: dp[i + 1][j + 1] = dp[i][j] + 1
        return max(item for row in dp for item in row)