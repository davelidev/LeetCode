class Answer(object):
'''133. Clone Graph'''
    def cloneGraph(node):
        start_node = node
        node_to_clone = {}
        def clone_if_not_exists(node):
            if node not in node_to_clone:
                node_to_clone[node] = UndirectedGraphNode(node.label)
        if not node:
            return None
        stack = [node]
        copied = set()
        while stack:
            node = stack.pop()
            if node not in copied:
                clone_if_not_exists(node)
                for neighbor in node.neighbors:
                    clone_if_not_exists(neighbor)
                    node_to_clone[node].neighbors.append(node_to_clone[neighbor])
                    stack.insert(0, neighbor)
                copied.add(node)
        return node_to_clone[start_node]