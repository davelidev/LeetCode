class Answer(object):'''796. Rotate String'''
    def rotateString(A, B):
        return not any([A, B]) or any(A[i:] + A[:i] == B for i in range(len(A)))