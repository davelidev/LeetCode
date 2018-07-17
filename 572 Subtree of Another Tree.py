class Answer(object):
'''572. Subtree of Another Tree'''
    def isSubtree(s, t):
        def serialize(node):
            if not node: return ''
            return '[%d,%s,%s]' %(node.val, serialize(node.left), serialize(node.right))
        return serialize(t) in serialize(s)