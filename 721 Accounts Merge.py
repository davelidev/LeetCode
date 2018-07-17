class Answer(object):
'''721. Accounts Merge'''
    def accountsMerge(self, accounts):
        graph = {}
        for acc in accounts:
            for i in range(1, len(acc)):
                graph.setdefault(acc[i], set())
                graph[acc[i]] |= set(acc[1:])
        to_be_visited = set(acc[i] for acc in accounts for i in range(1, len(acc)))
        def visite_node(node):
            if node in to_be_visited:
                to_be_visited.remove(node)
                return True
            return False
        
        email_to_name = {email:acc[0] for acc in accounts for email in acc[1:]}
        res = []
        while to_be_visited:
            start = to_be_visited.pop()
            visite_node(start)
            connected_nodes = []
            bfs = [start]
            while bfs:
                connected_nodes.extend(bfs)
                bfs = [adj for node in bfs for adj in graph[node] if visite_node(adj)]
            connected_nodes.sort()
            connected_nodes.insert(0, email_to_name[start])
            res.append(connected_nodes)
        return res