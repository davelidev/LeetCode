class Answer(object):
'''237. Delete Node in a Linked List'''
    def deleteNode(node):
        node.val = node.next.val
        node.next = node.next.next