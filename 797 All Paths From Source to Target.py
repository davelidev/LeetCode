class Answer(object):'''797. All Paths From Source to Target'''
    def allPathsSourceTarget(self, graph):
        paths, cur_path = [], []
        def dfs(cur):
            if cur == len(graph) - 1:
                paths.append(cur_path + [cur])
            else:
                cur_path.append(cur)
                for adj in graph[cur]: dfs(adj)
                cur_path.pop()
        dfs(0)
        return paths