class Answer(object):'''915. Partition Array into Disjoint Intervals'''
    def partitionDisjoint(self, A):
        r_min = [A[-1]]
        for i in range(len(A) -2, -1, -1):
            r_min.append(min(A[i], r_min[-1]))
        r_min.reverse()

        max_so_far = A[0]
        for i in range(len(A) - 1):
            if max_so_far <=  r_min[i + 1]:
                return i + 1
            max_so_far = max(max_so_far, A[i])
        return len(r_min)