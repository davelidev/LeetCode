class Answer(object):'''896. Monotonic Array'''
    def isMonotonic(A):
        if len(A) <= 2: return True
        return all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or                all(A[i] >= A[i + 1] for i in range(len(A) - 1))