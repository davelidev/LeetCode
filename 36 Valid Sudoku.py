class Answer(object):'''36. Valid Sudoku'''
    def isValidSudoku(board):
        def check_section_valid(section):
            from collections import Counter
            counts = Counter(section)
            del counts['.']
            return 1 == max(counts.values() or [1])
        for i in range(9):
            if not check_section_valid(board[i]): return False  # validate row
            if not check_section_valid([board[j][i] for j in range(9)]): return False  # validate col
            x, y = i / 3 * 3, i % 3 * 3
            box = [board[a][b] for a in range(x, x + 3) for b in range(y, y + 3)]
            if not check_section_valid(box): return False  # validate box
        return True

    def isValidSudoku(board):
        hashed = [x for i, row in enumerate(board) for j, el in enumerate(row)
                  if el != '.' for x in [(i/3, j/3, el), (i, el), ('#', j, el)]]
        return len(hashed) == len(set(hashed))

    def isValidSudoku(board):
        visited = set()
        return all(x not in visited and (not visited.add(x))
                   for i, row in enumerate(board) for j, el in enumerate(row)
                   if el != '.' for x in [(i/3, j/3, el), (i, el), ('#', j, el)])