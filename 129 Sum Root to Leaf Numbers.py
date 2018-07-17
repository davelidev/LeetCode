class Answer(object):
'''129. Sum Root to Leaf Numbers'''
    def sumNumbers(root):
        def _sumNumbers(node, cur_num=[]):
            if not node:
                return 0
            cur_num.append(node.val)
            if not node.left and not node.right:
                res = int(''.join(map(lambda x: str(x), cur_num)))
            else:
                res =  _sumNumbers(node.left, cur_num) + _sumNumbers(node.right, cur_num)
            cur_num.pop()
            return res
        return _sumNumbers(root)