class Answer(object):
'''399. Evaluate Division'''
    def calcEquation(equations, values, queries):
        graph = {}
        for i, (start, end) in enumerate(equations):
            graph.setdefault(start, {})[end] = float(values[i])
            graph.setdefault(end, {})[start] = 1 / float(values[i])

        def dist(start, end):
            if start not in graph: return -1
            bfs = [(start, 1)]
            visited = set()
            while bfs:
                end_node = next((val for node, val in bfs if node == end), None)
                if end_node is not None: return end_node
                bfs = [ (to_node, cur_val * graph[cur_node][to_node])
                            for cur_node, cur_val in bfs
                                for to_node in graph[cur_node]
                                    if to_node not in visited and (visited.add(to_node) is None)]
            return -1
        return [dist(start, end) for start, end in queries]