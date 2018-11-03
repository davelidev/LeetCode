class Answer(object):'''897. Increasing Order Search Tree'''
class Solution(object):
    def increasingBST(self, root):
        self.cur = dummy = TreeNode('dummy')
        def create_tree(node):
            if not node: return
            if node.left:
                create_tree(node.left)
            self.cur.right = TreeNode(node.val)
            self.cur = self.cur.right
            if node.right:
                create_tree(node.right)
        create_tree(root)
        return dummy.right