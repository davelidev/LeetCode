class Answer(object):
'''138. Copy List with Random Pointer'''
    # first method is to use hash map
    map all the nodes to a clone.
    iterate through each node
        map next_pointer => node_to_clone[node].next_node = node_to_clone[node.next_node]
        map random_pointr => node_to_clone[node].rand_node = node_to_clone[node.rand_node]

    # second method is to use new node as the mapper, interleave them in the nodes
    iterate through each node
        cloned_node = Node(node.val)
        cloned_node.next_node = node.next_node
        node.next_node = cloned_node
    # map the random pointer
    iterate through the even nodes(original nodes)
        node.next_node.rand_node = node.rand_node.rand_node
    head = cur_cloned_node = Node('Dummy')
    # map the next node
    cur = head
    while cur:
        cur_cloned_node.next_node = cur.next_node
        cur_cloned_node = cur_cloned_node.next_node
        cur.next_node = cur.next_node.next_node
        cur = cur.next_node