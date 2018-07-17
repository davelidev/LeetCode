class Answer(object):
'''207. Course Schedule'''
    def canFinish(numCourses, prerequisites):
        graph = [[] for _ in range(numCourses)]
        flow = [0] * numCourses
        bfs = set(range(numCourses))
        for course, preq in prerequisites:
            graph[preq].append(course)
            flow[course] += 1
            if course in bfs:
                bfs.remove(course)
        flow = map(lambda x: x if x else 1, flow)
        bfs = list(bfs)
        bfs2 = []
        while bfs:
            while bfs:
                next_node = bfs.pop()
                flow[next_node] -= 1
                if flow[next_node] == 0:
                    bfs2.extend(graph[next_node])
            bfs, bfs2 = bfs2, []
        return not any(flow)