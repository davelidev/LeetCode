class Answer(object):
'''210. Course Schedule II'''
    def findOrder(numCourses, prerequisites):
        graph = [set() for _ in range(numCourses)]
        flow_in = [0 for _ in range(numCourses)]
        for course, prereq in prerequisites:
            if course not in graph[prereq]:
                graph[prereq].add(course)
                flow_in[course] += 1
        bfs = [node for node, in_count in enumerate(flow_in) if in_count == 0]
        for node in bfs: flow_in[node] = 1
        res = []
        while bfs:
            adjs = []
            for node in bfs:
                flow_in[node] -= 1
                if not flow_in[node]:
                    res.append(node)
                    for to_node in graph[node]: adjs.append(to_node)
            bfs = adjs
        return res if len(res) == numCourses else []