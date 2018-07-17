class Answer(object):
'''655. Print Binary Tree'''
    def printTree(self, root):
        res = []
        bfs = [root]
        while any(bfs):
            res.append(bfs)
            bfs = [(kid if kid else '')                    for node in bfs for kid in (['', ''] if not node else [node.left, node.right])]
        padding = 0
        def convert_to_str(node): return str(node.val) if type(node) != str else ''
        def add_spaces(lst, num_spaces):
            for i in range(num_spaces): lst.append('')
        for i in range(len(res) - 1, -1, -1):
            spacing = padding * 2 + 1
            new_row = [""] * padding
            for j in range(len(res[i]) - 1):
                new_row.append(convert_to_str(res[i][j]))
                add_spaces(new_row, spacing)
            new_row.append(convert_to_str(res[i][-1]))
            add_spaces(new_row, padding)
            res[i] = new_row
            padding = spacing
        return res