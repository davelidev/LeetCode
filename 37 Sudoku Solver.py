class Answer(object):'''37. Sudoku Solver'''
    def solveSudoku(self, board):

        seen = set()

        def is_valid_add(i, j, el):
            seen_item = {(i, None, el), (None, j, el), (i/3, j/3, el)}
            if seen_item & seen: return False
            board[i][j] = el
            seen.update(seen_item)
            return True

        def remove_seen_item(i, j, el):
            for el in {(i, None, el), (None, j, el), (i/3, j/3, el)}:
                seen.remove(el)
            board[i][j] = '.'

        for i, col in enumerate(board):
            for j, el in enumerate(col):
                if el != '.': is_valid_add(i, j, el)  # add pre-existing

        def _solveSudoku(start_i):
            for i in range(start_i, 9 * 9):
                a, b = divmod(i, 9)
                if board[a][b] == '.':
                    for potential in map(str, range(1, 10)):
                        if is_valid_add(a, b, potential):
                            if _solveSudoku(i + 1): return True
                            remove_seen_item(a, b, potential)
                    return False
            return True
        _solveSudoku(0)