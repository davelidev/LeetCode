class Answer(object):
'''310. Minimum Height Trees'''
    def findMinHeightTrees(n, edges):
        graph = {i:[i] for i in range(n)}
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
        def _get_furthest(node):
            bfs = [node]
            furtest = None
            visited = set([node])
            while bfs:
                furtest = bfs[0]
                bfs = [adj for node in bfs for adj in graph[node] if adj not in visited and (not visited.add(adj))]
            return furtest
        def _get_midpoint(node1, node2):
            bfs1 = {node1}
            bfs2 = {node2}
            visited1, visited2 = set([node1]), set([node2])
            while bfs1:
                intersect = (bfs1 & bfs2) or ((bfs1 & visited2) | (bfs2 & visited1))
                if intersect: return list(intersect)
                bfs1 = {adj for node in bfs1 for adj in graph[node] if adj not in visited1 and (not visited1.add(adj))}
                bfs2 = {adj for node in bfs2 for adj in graph[node] if adj not in visited2 and (not visited2.add(adj))}        
        end1 = _get_furthest(0)
        end2 = _get_furthest(end1)
        return _get_midpoint(end1, end2)

    def findMinHeightTrees(n, edges):
        graph = {i:set() for i in range(n)}
        for a, b in edges:
            graph[a].add(b)
            graph[b].add(a)
        # iteratively remove leaves until 1/2 nodes left
        visited = set()
        while n - len(visited) > 2:
            leaves = [node for node, adjs in graph.iteritems() if len(adjs) == 1]
            for leave in leaves:
                for adj in graph[leave]:
                    if leave in graph[adj]: 
                        graph[adj].remove(leave)
                del graph[leave]
            visited.update(set(leaves))
        return list(set(range(n)) - visited)