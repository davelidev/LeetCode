class Answer(object):
'''684. Redundant Connection'''

    def findRedundantConnection(self, edges):
        from collections import defaultdict
        graph = defaultdict(set)
        for i, (a, b) in enumerate(edges):
            graph[a].add(b)
            graph[b].add(a)
            
        def _dfs_find_cycle_nodes(cur, prev=None, path=[], visited=set()):
            visited.add(cur)
            for adj in graph[cur]:
                if adj != prev:
                    path.append(cur)
                    if adj in visited: return path[path.index(adj):]
                    found = _dfs_find_cycle_nodes(adj, cur, path, visited)
                    if found: return found
                    path.pop()
        
        cycle = set(_dfs_find_cycle_nodes(edges[0][0]))
        return next(((a, b) for a, b in reversed(edges) if a in cycle and b in cycle), None)

    def findRedundantConnection(self, edges):
        # use union find + path compression to find a connected component
        node_to_parent = range(len(edges) + 1)
        def get_root(node):
            path, cur = set(), node
            while cur != node_to_parent[cur]:
                path.add(cur)
                cur = node_to_parent[cur]
            root = cur
            for node in path: node_to_parent[node] = root
            return root
        for a, b in edges:
            root_a, root_b = get_root(a), get_root(b)
            if root_a == root_b: return [a, b]
            node_to_parent[root_a] = root_b
        
        229. Majority Element II
    def majorityElement(self, nums):
        if not nums or len(nums) <= 1: return nums[:]

        candidate1, candidate2, counter1, counter2 = float('inf'), float('inf'), 0, 0
        for num in nums:
            if num == candidate1: counter1 += 1
            elif num == candidate2: counter2 += 1
            elif counter1 == 0: candidate1, counter1 = num, 1
            elif counter2 == 0: candidate2, counter2 = num, 1
            else:
                counter1 -= 1
                counter2 -= 1
        return [ i for i in [candidate1, candidate2] if nums.count(i) > (len(nums) / 3.)]