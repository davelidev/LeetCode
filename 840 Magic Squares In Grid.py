class Answer(object):
'''840. Magic Squares In Grid'''
    def numMagicSquaresInside(grid):
        def is_magic(i, j):
            qwe, asd, zxc = [grid[i + k][j:j + 3] for k in range(3)]
            (q,w,e), (a,s,d), (z,x,c) = qwe, asd, zxc
            qaz, wsx, edc = (q,a,z),(w,s,x),(e,d,c)
            qsc, esz = (q,s,c),(e,s,z)
            totals = map(sum,[qwe, asd, zxc, qaz, wsx, edc, qsc, esz])
            return {q,w,e,a,s,d,z,x,c} == {1,2,3,4,5,6,7,8,9} and                     all(totals[i] == totals[i - 1] for i in range(1, len(totals)))
        
        return sum(is_magic(i, j) for i in range(len(grid) - 2) for j in range(len(grid[0]) - 2))