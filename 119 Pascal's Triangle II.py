class Answer(object):
'''119. Pascal's Triangle II'''
    def getRow(rowIndex):
        if rowIndex <= 1: return [1] * (rowIndex + 1)
        cur_row = [1, 1]
        for i in range(rowIndex - 1):
            cur_row = [1] + [sum([cur_row[i-1], cur_row[i]]) for i in range(1, len(cur_row))] + [1]
        return cur_row