class Answer(object):'''6. ZigZag Conversion'''
    def convert(s, numRows):
        res = [[] for _ in range(numRows)]
        cur, direction = 0, 1
        for i, char in enumerate(s):
            res[cur].append(char)
            # change direction if the next idx is out of range
            direction *= 1 if (0 <= (direction + cur) < numRows) else -1
            cur += direction
        return ''.join([''.join(row) for row in res])