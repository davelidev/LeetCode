class Answer(object):
'''653. Two Sum IV - Input is a bfs'''
    def findTarget(root, k):
        if not root: return False
        visited = set()
        bfs = [root]
        while bfs:
            for node in bfs:
                if (k - node.val) in visited: return True
                visited.add(node.val)
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return False

    def findTarget(root, k):
        def flatten(node):
            if not node:
                return []
            new_lst = flatten(node.left)
            new_lst.append(node.val)
            new_lst.extend(flatten(node.right))
            return new_lst
        lst = flatten(root)
        i, j = 0, len(lst) - 1
        while i < j:
            i_j_sum = lst[i] + lst[j]
            if i_j_sum == k:
                return True
            elif i_j_sum < k:
                i += 1
            else: j -= 1
        return False