"# LeetCode" 
```python


'''149. Max Points on a Line'''
    def maxPoints(self, points):
        ''' O(V**2) time and O(V) space
        for each node, calculate the slope with adjacent nodes and keep track of the max
        edge cases => same coordinates, infinite slope, convert to float'''
        import numpy as np
        max_count = 0
        for i, point1 in enumerate(points):
            x1, y1 = point1.x, point1.y
            slope_cnt = {}
            same = 0
            for j, point2 in enumerate(points):
                if i != j:
                    x2, y2 = point2.x, point2.y
                    if (x1, y1) != (x2, y2):
                        slope = np.longdouble(y2 - y1) / (x2 - x1) if (x2 != x1) else 'inf'
                        slope_cnt[slope] = slope_cnt.get(slope, 1) + 1
                    else:
                        same += 1
            max_count = max(max(slope_cnt.values() or [1]) + same, max_count)
        return max_count or int(bool(points))

'''126. Word Ladder II'''
    def findLadders(beginWord, endWord, wordList):
        '''O(wl + V + E) time, O(wl * V + E) space, where wl is the max word length
        # Use bfs to find the closest path, words with one letter difference form a path
        initialize and construct the edges like the following:
            for words abc, abd => {'_bc':['abc'], 'a_c':['abc'],'ab_':['abc', 'abd'],'_bd':['abd'],'a_d':['abd']}
            where the keys are the words with 1 character hidden, values are list of words that match the pattern.
            The keys are the nodes, values are the edges
        use bfs for seaching from start to end word
        each node is a word, the adj nodes(hide each of the characters in the word to get all the adj words)'''
        from collections import defaultdict
        n = len(beginWord)
        adj_dict = defaultdict(list)
        for word in wordList:
            for i in range(n):
                adj_dict[word[:i] + '?' + word[i + 1:]].append(word)
        bfs = [([beginWord], beginWord)]
        visited = {beginWord}
        while bfs:
            lvl_visited = set()
            bfs = [(path + [adj], adj)
                   for path, cur in bfs
                   for i in range(n)
                   for adj in adj_dict[cur[:i] + '?' + cur[i + 1:]]
                   if adj not in visited and (lvl_visited.add(adj) is None)]
            visited.update(lvl_visited)
            final_path = [path for path, word in bfs if word == endWord]
            if final_path: return final_path
        return []

'''132. Palindrome Partitioning II'''
    def minCut(s):
        n = len(s)
        is_pal = [[None] * (n + 1) for _ in range(n + 1)]
        for i in range(n): is_pal[i][i] = is_pal[i][i + 1] = True
        for k in range(2, n + 1):
            for i in range(n - k + 1):
                j = i + k
                is_pal[i][j] = s[i] == s[j - 1] and is_pal[i + 1][j - 1]
        dp = [float('inf')] * (n + 1)
        dp[0] = -1
        for j in range(n + 1):
            if j: dp[j] = dp[j - 1] + 1
            for i in range(j):
                if is_pal[i][j]: dp[j] = min(dp[j], dp[i] + 1)
        return dp[-1]

'''212. Word Search II'''
    
    def findWords(board, words):
        '''build a trie-tree and use back tracing that starts from each coordinate 
        build the trie tree using dictionary, for words abc, adc, abd, it will look like this
        {'a':{'b':{'c':{'word':None},'d':{'word':None}},'d':{'c':{'word':None}}}}'''
        m, n = len(board), len(board[0])
        tree = {}
        for word in words:
            cur = tree
            for char in word:
                cur = cur.setdefault(char, {})
            cur[True] = True
        seen = set()
        word = []
        def dfs(i, j, cur):
            word.append(board[i][j])
            if board[i][j] in cur:
                if True in cur[board[i][j]]: seen.add(''.join(word))
                tmp, board[i][j] = board[i][j], None

                for x, y in [(i + dx, j + dy)
                             for dx, dy in zip([1,0,-1,0], [0,1,0,-1])
                             if 0 <= i + dx < m and 0 <= j + dy < n]:
                    if board[x][y]:
                        dfs(x, y, cur[tmp])
                board[i][j] = tmp
            word.pop()
        for i in range(m):
            for j in range(n):
                dfs(i, j, tree)
        return list(seen)

'''68. Text Justification'''
    def fullJustify(words, maxWidth):
        '''Use a queue. For each line, add words to the queue, including spaces as delimitor,
         while keeping count of number of chars. If the queue overflows,
          then add to the result and empty the line in the queue.
            O(n) time, O(ll) space , n is number of words, ll is length of line'''
        sentences = []
        num_char, sent_words = 0, []
        for word in words:
            num_char += len(word)
            sent_words.append(word)
            if num_char + len(sent_words) - 1 > maxWidth:
                num_char -= len(word)
                sent_words.pop()
                if len(sent_words) == 1: sent_words.append('') # make left justifiy if only one word
                avg_space, overflow_spaces = divmod((maxWidth - num_char), len(sent_words) - 1)
                for i in range(1, overflow_spaces + 1): sent_words[i] = ' ' + sent_words[i]
                sentences.append((' ' * avg_space).join(sent_words))
                num_char, sent_words = len(word), [word]
        sentences.append(' '.join(sent_words).ljust(maxWidth))
        return sentences

'''76. Minimum Window Substring'''
    def minWindow(s, t):
        from collections import Counter
        start = 0
        t_counts = Counter(t)
        missing_count = len(t_counts)
        i, j = 0, float('inf')
        for end, c in enumerate(s):
            t_counts[c] -= 1
            if t_counts[c] == 0: missing_count -= 1
            while missing_count == 0:
                if end - start < j - i:
                    i, j = start, end + 1
                t_counts[s[start]] += 1
                if t_counts[s[start]] == 1: missing_count += 1
                start += 1
        return s[i: j] if j != float('inf') else ''

'''45. Jump Game II'''
    def jump(nums):
        if len(nums) == 1:
            return 0
        step_number = 0

        cur_max = nums[0]
        to_visit = (1, 1 + cur_max)

        while True:
            new_max = to_visit[1]
            step_number += 1
            if new_max >= len(nums):
                return step_number
            for i in range(to_visit[0], to_visit[1]):
                new_max = max(new_max, nums[i] + i + 1)
            to_visit = (to_visit[1], new_max)

'''630. Course Schedule III'''
    def scheduleCourse(courses):
        import heapq
        courses = sorted([(deadline, dur) for dur, deadline in courses])
        max_heap = []
        max_len = total_dur = 0
        for deadline, dur in courses:
            total_dur += dur
            heapq.heappush(max_heap, -dur)
            while total_dur > deadline:
                total_dur += heapq.heappop(max_heap)
            max_len = max(max_len, len(max_heap))
        return max_len

'''84. Largest Rectangle in Histogram'''
    def largestRectangleArea(heights):
        heights.append(0)
        dp = []
        max_area = 0
        for i, height in enumerate(heights):
            left = i
            while dp and dp[-1][1] > height:
                left = dp[-1][0]
                j, j_height = dp.pop()
                max_area = max(max_area, j_height * (i - j))
            dp.append((left, height))
        return max_area

'''41. First Missing Positive'''
    def firstMissingPositive(nums):
        # index visited is marked as negative
        # pass1: make negative nums positive, and make it len(nums) + 1 so it does not influence our algorithm
        # pass2: mark all the indicies visited as negative
        # pass3: find the first positive
        for i, num in enumerate(nums):
            if num <= 0: nums[i] = len(nums) + 1
        nums.append(len(nums) + 1)
        for i, num in enumerate(nums):
            if 0 < abs(num) < len(nums): nums[abs(num)] = -abs(nums[abs(num)])
        return next((i for i, num in enumerate(nums) if i and num > 0), len(nums))

'''51. N-Queens'''
    def solveNQueens(n):
        vertical, diag1, diag2 = [[False] * (2 * n) for _ in range(3)]
        cur = [['.'] * n for _ in range(n)]
        res = []
        def _solveNQueens(y):
            if y >= n: return
            for x in range(0, n):
                diag1_i, diag2_i = (x - y) % (2 * n), x + y
                if not vertical[x] and not diag1[diag1_i] and not diag2[diag2_i]:
                    cur[x][y] = 'Q'
                    vertical[x] = diag1[diag1_i] = diag2[diag2_i] = True
                    if y == n - 1: res.append([''.join(row) for row in cur])
                    else: _solveNQueens(y + 1)
                    vertical[x] = diag1[diag1_i] = diag2[diag2_i] = False
                    cur[x][y] = '.'
        _solveNQueens(0)
        return res

'''87. Scramble String'''
    def isScramble(s1, s2):
        dp = {}
        from collections import Counter
        def _isScramble(s1, s2):
            if s1 == s2: return True
            elif (s1, s2) in dp: return dp[(s1, s2)]
            elif sorted(s1) != sorted(s2):
                dp[s1, s2] = False
                return False
            n, f = len(s1), _isScramble
            for i in range(1, len(s1)):
                if f(s1[i:], s2[i:]) and f(s1[:i], s2[:i]) or \
                   f(s1[i:], s2[:-i]) and f(s1[:i], s2[-i:]):
                    dp[(s1, s2)] = True
                    return True
            dp[(s1, s2)] = False
            return False
        return _isScramble(s1, s2)

'''224. Basic Calculator'''
    def calculate(s):
        s = list(('((%s)'%s).replace(' ', ''))
        nested_brackets, num = [], []
        for i, c in enumerate(s):
            if c == '(':
                nested_brackets.append([0, '+'])
                num = []
            elif c == '-' and s[i-1] in '+-(':
                prev_sign = nested_brackets[-1][1]
                nested_brackets[-1][1] = '+' if prev_sign == '-' else '+'
            elif c in '+-)':
                if num:
                    num = int(''.join(num))
                    sign = (1 if nested_brackets[-1][1] == '+' else -1)
                    nested_brackets[-1][0] += sign * num
                    num = []
                    if c == ')':
                        num = list(str(nested_brackets[-1][0]))
                        nested_brackets.pop()
                    else: nested_brackets[-1][1] = c
            elif c.isdigit(): num.append(c)
        return int(''.join(num))

'''23. Merge k Sorted Lists'''
    def mergeKLists(lists):
        import heapq
        heap = [(lst.val, lst) for lst in lists if lst]
        heapq.heapify(heap)
        cur = dummy_head = ListNode('dummy')
        while heap:
            elem, lst = heapq.heappop(heap)
            cur.next = lst
            cur, lst = cur.next, lst.next
            if lst: heapq.heappush(heap, (lst.val, lst))
        return dummy_head.next

'''391. Perfect Rectangle'''
    # hash all the corners, check count to ensure all corners have 2/4 count, and the outer corners have exactly 1 count
    def isRectangleCover(rectangles):

        def get_corners(bottom_left, top_right):
            return [tuple(top_right), tuple(bottom_left),
                    (top_right[0], bottom_left[1]), (bottom_left[0], top_right[1])]

        def get_area(bottom_left, top_right):
            return (top_right[1] - bottom_left[1]) * (top_right[0] - bottom_left[0])

        from collections import defaultdict
        hashed_count = defaultdict(int)
        top_right, bottom_left = (float('-inf'), ), (float('inf'), )
        area_by_acc = 0

        for rect in rectangles:
            top_right = max(top_right, tuple(rect[2:]))
            bottom_left = min(bottom_left, tuple(rect[:2]))
            corners = get_corners(rect[:2], rect[2:])
            area_by_acc += get_area(rect[:2], rect[2:])
            for corner in corners: hashed_count[corner] += 1

        counts = hashed_count.values()
        outer_corners_by_1_count = set(corner for corner, count in hashed_count.iteritems() if count == 1)
        outer_corners_by_loop = set(get_corners(bottom_left, top_right))

        area_by_outer_corners = get_area(bottom_left, top_right)
    
        return len(outer_corners_by_1_count) == 4 and \
            outer_corners_by_1_count == outer_corners_by_loop and \
            area_by_acc == area_by_outer_corners and \
            next((False for count in counts if count % 2 != 0 and count != 1), True) \

'''295. Find Median from Data Stream'''
    # maintain 2 heaps, one is min heap for storing the max half of the numbers, one is max heap for storing min half of the numbers. The maximun number is generated if push and poll from one of the heaps, then push into another.
    class MedianFinder(object):
        def __init__(self):
            self.larger = MinHeap()
            self.smaller = MaxHeap()
        def addNum(self, num):
            if len(self.smaller) == len(self.larger):
                self.smaller.push(self.larger.push(num).poll())
            else:
                self.larger.push(self.smaller.push(num).poll())
        def findMedian(self):
            if 0 < len(self.smaller) == len(self.larger):
                return (self.smaller.seek() + self.larger.seek())/2
            return self.smaller.seek()

'''480. Sliding Window Median'''
    # a variation of "295. Find Median from Data Stream", with 1 additional function, remove the a specific element from the heap(can be done using hashmap). At each sliding window, remove the element on the left and add the element on the right.

'''57. Insert Interval'''
        def insert(intervals, newInterval):
        left, right = [], []
        s, e = newInterval.start, newInterval.end
        for interval in intervals:
            if interval.end < s: left.append(interval)
            elif e < interval.start: right.append(interval)
            else: s, e = min(interval.start, s), max(interval.end, e)
        return left + [Interval(s, e)] + right

'''675. Cut Off Trees for Golf Event'''
    # First iterate through the forest and store the values along with their associated positions. Sort the values and iterate through the list to cut off the trees one by one while accumulating the steps taken. Searching can be done using bfs with a visited algorithm.
    def cutOffTree(forest):
        m, n = len(forest), len(forest[0]) if forest else 0
        search = sorted([(forest[i][j], i, j)
                         for i in range(m)
                         for j in range(n)
                         if forest[i][j] > 1])

        xy_dir = zip([1, 0, -1, 0], [0, 1, 0, -1])
        def search_tree(x1, y1, x2, y2, height):
            visited = [([None] * n) for i in range(m)]
            def get_adjs(x, y):
                return [(a, b)
                        for a, b in [(x + x_d, y + y_d) for x_d, y_d in xy_dir]
                        if 0 <= a < m and 0 <= b < n \
                        and 0 < forest[a][b]]
            bfss = [[(x1, y1)], [(x2, y2)]]
            if (x1, y1) == (x2, y2): return 0
            visited[x1][y1] = 0
            visited[x2][y2] = 1
            step = 0
            while all(bfss):
                new_bfs = []
                bfs_i = step % 2
                for x, y in bfss[bfs_i]:
                    for a_x, a_y in get_adjs(x, y):
                        if visited[a_x][a_y] is None:
                            new_bfs.append((a_x, a_y))
                            visited[a_x][a_y] = bfs_i
                        elif visited[a_x][a_y] != bfs_i: return step + 1
                step += 1
                bfss[bfs_i] = new_bfs
        
        prev_x, prev_y = 0, 0
        total_steps = 0
        for tree in search:
            height, x, y = tree
            steps = search_tree(prev_x, prev_y, x, y, height)
            prev_x, prev_y = x, y
            if steps is None: return -1
            total_steps += steps
        return total_steps
        

'''124. Binary Tree Maximum Path Sum'''
    # Recursion, each returns 2 values, first one without going through the root, the second one goes through the root. take the max, and return max(left_no_root, right_no_root, root + left_root + right_root)
    def maxPathSum(root):
        def _maxPathSum(node):
            if not node: return float('-inf'), float('-inf')
            (l_i, l_x), (r_i, r_x) = _maxPathSum(node.left), _maxPathSum(node.right)
            inc = max(l_i, r_i, 0) + node.val
            exc = max(l_x, r_x, max(l_i, 0) + max(r_i, 0) + node.val)
            return inc, exc
        
        return max(_maxPathSum(root))

'''403. Frog Jump'''
    def canCross(stones):
        from collections import defaultdict
        stone_to_steps = defaultdict(set)
        if (stones[1] - stones[0]) != 1: return False
        stone_to_steps[stones[1]].add(1)
        for pos in stones:
            for step in stone_to_steps[pos]:
                for new_pos in [pos + step + i for i in [-1, 0, 1]]:
                    if new_pos == stones[-1]: return True
                    elif new_pos != pos:
                        stone_to_steps[new_pos].add(new_pos - pos)
        return False 
                        

'''42. Trapping Rain Water'''
    #Have 2 pointers, one from the left, one from the right. Keep track of the max from the left as well as the max from the right. The water level in the current block is calculated by min(left_max, right_max) - cur_ground. move i from the left to right if max_left < max_right, move j fromt he right to left  otherwise.

'''301. Remove Invalid Parentheses'''
    def check_valid(lst):
        level = 0
        for item in lst:
            if level < 0: return False
            elif item == '(': level += 1
            elif item == ')': level -= 1
            elif level == 0: return Fale
        return level == 0
    lst = []
    for char in input_lst:
        if char == '(' or char == ')': lst.append(char)
        elif lst and lst[-1] != '(' and lst[-1] != ')': lst[-1] += char
        else: lst.append(char)
    bfs = [lst]
    bfs_secondary = []
    res = []
    while not res:
        while bfs:
            next_elem = bfs.pop()[:]
            if check_valid(next_elem): res.append(next_elem)
            elif not res:
                for i in range(len(next_elem)):
                    if next_elem[i] == '(' or next_elem[i] == ')':
                        bfs_secondary.append(next_elem[:i] + next_elem[i + 1:])
        bfs = bfs_secondary
        bfs_secondary = []

'''145. Binary Tree Postorder Traversal'''
    def postorderTraversal(root):
        if not root: return []
        bfs = [root]
        while any(None if type(node) != TreeNode else node for node in bfs):
            bfs = [kid for node in bfs
                   for kid in ([node.left, node.right, node.val] if type(node) == TreeNode else [node]) if kid is not None]
        return bfs

'''128. Longest Consecutive Sequence'''
    def longestConsecutive(nums):
        consecutive = {}
        max_size = 0
        for num in nums:
            if num not in consecutive:
                size = 1
                left = right = None
                if num - 1 in consecutive:
                    size += consecutive[num - 1]
                    left = (num - 1)-(consecutive[num - 1] - 1)
                if num + 1 in consecutive:
                    size += consecutive[num + 1]
                    right = (num + 1) + (consecutive[num + 1] - 1)
                    consecutive[right] = size
                if left is not None:
                    consecutive[left] = size
                consecutive[num] = size
                max_size = max(max_size, size)
        return max_size

'''632. Smallest Range'''
    def smallest_range(lst_of_lsts):
        min_heap = MinHeap()
        greedy_iter = []
        for i, lst in enumerate(lst_of_lsts):
            if lst:
                min_heap.push([lst[i][0], [i, 0]])
                greedy_iter.append(lst[i][0])
            else:
                return
        shotest_dist = max(greedy_iter) - min(greedy_iter)
        while min_heap:
            val, coord = min_heap.pop()
            greedy_iter[coord[0]] = lst_of_lsts[coord[0]][coord[1]]
            shotest_dist = max(greedy_iter) - min(greedy_iter)
            if coord[1] + 1 >= len(lst_of_lsts[coord[0]]):
                return shotest_dist
            min_heap.push([lst[coord[0]][coord[1] + 1], [coord[0], coord[1] + 1]])

'''329. Longest Increasing Path in a Matrix'''
    def longestIncreasingPath(matrix):
        xy_dir = zip([-1, 0, 1, 0], [0, -1, 0, 1])
        m, n = len(matrix), len(matrix[0]) if matrix else 0
        dp = [[None] * n for _ in range(m)]
        def get_max_inc(x, y):
            if dp[x][y] is not None: return dp[x][y]
            adj = [(i, j) for i, j in [(x + x_d, y + y_d) for x_d, y_d in xy_dir]
                   if 0 <= i < m and 0 <= j < n and matrix[x][y] > matrix[i][j]]
            dp[x][y] = max([(get_max_inc(i, j) + 1) for i, j in adj] or [1])
            return dp[x][y]
        return max([get_max_inc(x, y) for x in range(m) for y in range(n)] or [0])

'''352. Data Stream as Disjoint Intervals'''
    # This is a variation of "128. Longest Consecutive Sequence". In addition, keep a list of start times to reconstruct at later time.
    class SummaryRanges(object):
        def __init__(self):
            self.start_times = set()
            self.start_time_to_num_of_elems = {}
        def addNum(self, num):
            consecutive = self.start_time_to_num_of_elems
            start_times = self.start_times
            if num not in consecutive:
                size = 1
                left = right = None
                if num - 1 in consecutive:
                    size += consecutive[num - 1]
                    left = (num - 1) - (consecutive[num - 1] - 1)
                if num + 1 in consecutive:
                    start_times.remove(num + 1)
                    size += consecutive[num + 1]
                    right = (num + 1) + (consecutive[num + 1] - 1)
                    consecutive[right] = size
                if left is not None:
                    consecutive[left] = size
                    start_times.add(left)
                else:
                    start_times.add(num)
                consecutive[num] = size
        def getIntervals(self):
            res = []
            for start_time in sorted(self.start_times):
                res.append([start_time, start_time + self.start_time_to_num_of_elems[start_time] - 1])
            return res if res else None

'''52. N-Queens II'''
    def totalNQueens(n):
        vertical, diag1, diag2 = [[False] * (2 * n) for _ in range(3)]
        self.count = 0
        def _totalNQueens(y):
            if y >= n: return
            for x in range(0, n):
                diag1_i, diag2_i = (x - y) % (2 * n), x + y
                if not vertical[x] and not diag1[diag1_i] and not diag2[diag2_i]:
                    vertical[x] = diag1[diag1_i] = diag2[diag2_i] = True
                    if y == n - 1: self.count += 1
                    else: _totalNQueens(y + 1)
                    vertical[x] = diag1[diag1_i] = diag2[diag2_i] = False
        _totalNQueens(0)
        return self.count

'''381. Insert Delete GetRandom O(1) - Duplicates allowed'''
    # need to edit the bellow code to allow for duplicates, assume no collision
    from random import randint
    class RandomizedCollection(object):
        def __init__(self, size=1028):
            self.size = size
            self.idx_vals_pair = [None] * size
            self.idx_mapper = []

        def hash_algo(self, val):
            return hash(val)  # assume there's more sophisticated method to do so

        def insert(self, val):
            self.idx_vals_pair[self.hash_algo(val) % self.size] = [len(self.idx_mapper), val]
            self.idx_mapper.append(self.hash_algo(val) % self.size)

        def remove(self, val):
            idx = self.idx_vals_pair[self.hash_algo(val) % self.size][0]
            self.idx_vals_pair[self.hash_algo(val) % self.size] = None
            self.idx_mapper[idx] = None
            if len(self.idx_mapper) == 1:
                self.idx_mapper.pop()
            else:
                new_idx = self.idx_mapper.pop()
                if not new_idx:
                    return
                self.idx_vals_pair[new_idx][0] = idx
                self.idx_mapper[idx] = new_idx

        def getRandom(self):
            if not self.idx_mapper:
                return None
            return self.idx_vals_pair[self.idx_mapper[randint(0, len(self.idx_mapper) - 1)]][1]

'''432. All O`one Data Structure'''
    # create a min heap and max heap. For inc, if the key exists, search, increment and modify its position accordingly in both heaps, otherwise insert it to both heaps. For dec, if the key exists, search, decrement and modify its position accordingly in both heaps. getMaxKey, and getMinKey is simply a call to the heap function to get the maximum and minimum.

'''468. Validate IP Address'''
    def validIPAddress(IP):
        def is_hex(s):
            hex_digits = set("0123456789abcdefABCDEF")
            for char in s:
                if not (char in hex_digits):
                    return False
            return True
        if '.' in IP:
            IP = IP.split('.')
            if len(IP) != 4:
                return "Neither"
            for ip in IP:
                try:
                    ip_int = int(ip)
                    if ip_int > 255 or ip_int < 0 or str(ip_int) != ip:
                        return "Neither"
                except:
                    return "Neither"
            return 'IPv4'
        elif ':' in IP:
            IP = IP.split(':')
            if len(IP) != 8:
                return "Neither"
            for ip in IP:
                if len(ip) > 4 or len(ip) == 0 or not is_hex(ip):
                    return 'Neither'
            return 'IPv6'
        return "Neither"
            
            

'''3. Longest Substring Without Repeating Characters'''
    keep a set of visited chars, iterate through the string. Keep count of number of visited chars, update it in each iteration. If the char is already visited, remove from the set and reset the count.
    O(n) time, O(n) space

'''165. Compare Version Numbers'''
    # split by '.' and compare from left to right one by one

'''98. Validate Binary Search Tree'''
    def isValidBST(root):
        def _isValidBST(node, min_val, max_val):
            if not node: return True
            if min_val < node.val < max_val and _isValidBST(node.left, min_val, node.val) and _isValidBST(node.right, node.val, max_val):
                return True
            return False
        return _isValidBST(root, float('-inf'), float('inf'))

'''523. Continuous Subarray Sum'''
    sum_so_far = 0
    prev_sums = set()
    for i in range(len(lst) - 1, -1, -1):
        sum_so_far += lst[i]
        if (k - (sum_so_far % k)) in prev_sums:
            return True
        prev_sums.add(sum_so_far % k)
    return False

'''307. Range Sum Query - Mutable'''
    class NumArray(object):
        def __init__(self, nums):
            self.update = nums.__setitem__
            self.sumRange = lambda i, j: sum(nums[i:j+1])

'''61. Rotate List'''
    def rotateRight(head, k):
        if not head:
            return
        count = 0
        cur = head
        tail = None
        while cur:
            count += 1
            tail, cur = cur, cur.next
        k = k % count
        if not k:
            return head
        cur = head
        for i in range(count - k - 1):
            cur = cur.next
        head, cur.next, tail.next = cur.next, None, head
        return head

'''179. Largest Number'''
    def largestNumber(nums):
        comp_func = lambda x, y : 1 if str(x) + str(y) > str(y) + str(x) else -1
        return str(int(''.join(sorted(map(lambda x: str(x), nums), reverse=True, cmp=comp_func))))

'''15. 3Sum'''
    def threeSum(nums):
        nums.sort()
        res = []
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            i_num = nums[i]
            j = i + 1
            k = len(nums) - 1
            while j < k:
                j_num = nums[j]
                k_num = nums[k]
                total = i_num + j_num + k_num
                if total == 0:
                    res.append([i_num, j_num, k_num])
                    while j < k and nums[j] == nums[j + 1]:
                        j += 1
                    while j < k and nums[k] == nums[k - 1]:
                        k -= 1
                    j += 1
                elif total > 0:
                    k -= 1
                elif total < 0:
                    j += 1
        return res

'''304. Range Sum Query 2D - Immutable'''
    class NumMatrix(object):

        def __init__(self, matrix):
            self.matrix = matrix
            for i in range(len(matrix)):
                for j in range(len(matrix[0]) if matrix else 0):
                    matrix[i][j] = self._sub_points([[i - 1, j], [i, j - 1], [i, j]], [[i - 1, j - 1]])
        
        def _sub_points(self, coords, sub_coords):
            def sum_points(coords):
                return sum(self.matrix[x][y] for x, y in coords if x >=0 and y >= 0)
            return sum_points(coords) - sum_points(sub_coords)
                    
        def sumRegion(self, row1, col1, row2, col2):
            return self._sub_points([[row2, col2], [row1 - 1, col1 - 1]],[[row1 - 1, col2], [row2, col1 - 1]])

'''71. Simplify Path'''
    simplified_path = []
    for folder in path.split('/'):
        if folder == '..':
            simplified_path.pop()
        else:
            simplified_path.append(folder)
    simplified_path = '/'.join(simplified_path)

'''322. Coin Change'''
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, len(dp)):
        for coin in coins:
            prev_idx = i - coin
            if prev_idx >= 0:
                dp[i] = min(dp[prev_idx] + 1, dp[i])
    res = dp[amount] if type(dp[amount]) == int else -1

'''152. Maximum Product Subarray'''
    def maxProduct(nums):
        large = small = max_val = nums[0]
        for i in range(1, len(nums)):
            num = nums[i]
            vals = [num, small * num, large * num]
            small, large = min(vals), max(vals)
            max_val = max(large, max_val)
        return max_val

'''324. Wiggle Sort II'''
    # Iterate through each number from left to right of a b c d e f. If a > b, then it's out of order, since b is in even position, and b is less than the previous number a, then swap a and b inplace, otherwise continue. Do the same for even position elements.
    # O(n) time, O(1) space

'''18. 4Sum'''
    def fourSum(nums, target):
        n = len(nums)
        res = set()
        from collections import defaultdict
        sum_to_ind = defaultdict(list)
        for i in range(2, n):
            for j in range(i + 1, n):
                sum_to_ind[(nums[i] + nums[j])].append((i, [nums[i], nums[j]]))
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                pair_1 = [nums[i], nums[j]]
                pair_1_sum = sum(pair_1)
                new_tar = target - pair_1_sum
                if new_tar in sum_to_ind:
                    for idx, pair_2 in reversed(sum_to_ind[new_tar]):
                        if idx <= j: break
                        res.add(tuple(sorted(pair_1 + pair_2)))
        return list(res)

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

'''130. Surrounded Regions'''
    # Go through all the edge regions, if the region is a part of an island, then mark and convert the island to '~'. Go though all the regions, and convert 'O' and 'X' as well as convert '~' to 'O'.

    def convert(i, j, from_sym, to_sym):
        if not ((0 <= i < len(board)) and 0 <= j < len(board[0])):
            return
        if board[i][j] == from_sym:
            board[i][j] = to_sym
            convert(i - 1, j, from_sym, to_sym)
            convert(i + 1, j, from_sym, to_sym)
            convert(i, j - 1, from_sym, to_sym)
            convert(i, j + 1, from_sym, to_sym)

    for i in range(len(board)):
        convert(i, 0, 'O', '~')
        convert(i, len(board[0]) - 1, 'O', '~')
    for j in range(len(board[0])):
        convert(0, j, 'O', '~')
        convert(len(board), j, 'O', '~')
    for i in range(len(board)):
        for j in range(len(board[0])):
            convert(len(board), j, 'O', 'X')
            convert(len(board), j, '~', 'O')

'''127. Word Ladder'''
    def ladderLength(beginWord, endWord, wordList):
        from collections import defaultdict
        n = len(beginWord)
        adj_dict = defaultdict(list)
        for word in wordList:
            for i in range(n):
                adj_dict[word[:i] + '?' + word[i + 1:]].append(word)
        bfs = [(1, beginWord)]
        visited = {beginWord}
        while bfs:
            bfs = [
                (count + 1, adj)
                for count, cur in bfs
                for i in range(n)
                for adj in adj_dict[cur[:i] + '?' + cur[i + 1:]]
                if adj not in visited and (visited.add(adj) is None)
            ]
            final_path = next((count for count, word in bfs if word == endWord), None)
            if final_path: return final_path
        return 0

'''402. Remove K Digits'''
    num = [int(x) for x in list(num)]
    i = 1
    while i < len(num) and k != 0:
        if num[i] > num[i - 1]:
            num.pop(i - 1)
            k -= 1
        else:
            i += 1
    num = ''.join([str(dig) for dig in num])

'''5. Longest Palindromic Substring'''
    def longestPalindrome(s):
        def get_pal(i, j):
            while 0 < i and j  < len(s) - 1 and s[i - 1] == s[j + 1]:
                i -= 1
                j += 1
            return [i, j + 1]
                
        max_pal = ""
        for idx in range(len(s)):
            pal1 = get_pal(idx, idx)
            pal2 = get_pal(idx + 1, idx)
            if pal1[1]-pal1[0] > len(max_pal):
                max_pal = s[pal1[0]: pal1[1]]
            if pal2[1] - pal2[0] > len(max_pal):
                max_pal = s[pal2[0]: pal2[1]]
        return max_pal

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

'''222. Count Complete Tree Nodes'''
    def countNodes(root):
        def _countNodes(node):
            left_node, left_depth = node, 0
            while left_node: left_node, left_depth = left_node.left, left_depth + 1
            right_node, right_depth = node, 0
            while right_node: right_node, right_depth = right_node.right, right_depth + 1
            if left_depth == right_depth: return 2 ** left_depth - 1
            else: return _countNodes(node.left) + 1 + _countNodes(node.right)
        return _countNodes(root)

'''151. Reverse Words in a String'''
    " ".join(filter(lambda x: x != "", s.split(" "))[::-1])

'''54. Spiral Matrix'''
    def spiralOrder(matrix):
        if not matrix:
            return []
        i = 0
        res = []
        a, b, c, d = 0, len(matrix[0]) - 1, len(matrix) - 1, 0
        while a <= c and b >= d:
            if i % 4 == 0:
                for j in range(d, b + 1):
                    res.append(matrix[a][j])
                a += 1
            elif i % 4 == 1:
                for j in range(a, c + 1):
                    res.append(matrix[j][b])
                b -= 1
            elif i % 4 == 2:
                for j in range(b, d - 1, -1):
                    res.append(matrix[c][j])
                c -= 1
            elif i % 4 == 3:
                for j in range(c, a - 1, -1):
                    res.append(matrix[j][d])
                d += 1
            i += 1
        return res

'''306. Additive Number'''
    #Use a for loop with i, j, k where i < j < k to loop though string. Find all the possible combinations with prefix that begin with an additive sequence. For each possible combination, check whether or not the whole string is a additive sequence.

'''8. String to Integer (atoi)'''
    def myAtoi(s):
        s = s.lstrip(' ')
        sign = 1
        if s.startswith('-'): s, sign = s[1:], -1
        elif s.startswith('+'): s = s[1:]
        s = s[:next((i for i, num in enumerate(s) if not num.isdigit()), len(s))]
        if not s: return 0
        int_rep = reduce(lambda x, y: x * 10 + (ord(y) - ord('0')), s, 0)
        return max(min(int_rep * sign, 2147483647), -2147483648)

'''150. Evaluate Reverse Polish Notation'''
    def evalRPN(tokens):
        stack = []
        str_to_expr = {
            '+':lambda x, y: x + y,
            '-':lambda x, y: x - y,
            '*':lambda x, y: x * y,
            '/':lambda x, y: x / y
        }
        for char in tokens:
            if char in '+-*/':
                a, b = stack.pop(), stack.pop()
                stack.append(str_to_expr[char](b, a))
            else:
                stack.append(int(char))
        return stack[0]

'''79. Word Search'''
    A variation of "212. Word Search II"

'''105. Construct Binary Tree from Preorder and Inorder Traversal'''
    def buildTree(preorder, inorder):
        def _buildTree(pre_s, pre_e, in_s, in_e):
            if pre_s >= pre_e or in_s >= in_e:
                return
            root = TreeNode(preorder[pre_s])
            idx = inorder.index(root.val, in_s, in_e)
            left_dist = idx - in_s
            root.left = _buildTree(pre_s + 1, pre_s + 1 + left_dist, in_s, idx)
            right_dist = in_e - idx - 1
            root.right = _buildTree(pre_s + 1 + left_dist, pre_s + 1 + left_dist + right_dist, idx + 1, idx + 1 + right_dist)
            return root
        return _buildTree(0, len(preorder), 0, len(preorder))
        86. Partition List
        Definition for singly-linked list.
    class ListNode(object):
        def __init__(self, x):
            self.val = x
            self.next = None
    def partition(head, x):
        left = ListNode('Dummy')
        right = ListNode('Dummy')
        left_cur, right_cur = left, right
        cur = head
        while cur:
            if cur.val <= x:
                left_cur.next = cur
                left_cur = left_cur.next
            else:
                right_cur.next = cur
                right_cur = right_cur.next
            cur = cur.next
        left_cur.next = right.next
        return left.next

'''134. Gas Station'''
    def canCompleteCircuit(gas, cost):
        diff = [gas[i] - cost[i] for i in range(len(gas))]
        if len(diff) == 1:
            if diff[0] >= 0:
                return 0
            else:
                return -1
        start_idx = 0
        end_idx = 0
        if not diff:
            return True
        # keep acc positive
        acc = diff[start_idx]
        for i in range(len(diff) - 1):
            if acc <= 0:
                start_idx -= 1
                acc += diff[start_idx % len(diff)]
            else:
                end_idx += 1
                acc += diff[end_idx % len(diff)]
        if acc >= 0:
            return start_idx % len(diff)
        else:
            return -1

'''221. Maximal Square'''
    def maximalSquare(matrix):
        from itertools import chain
        m, n = len(matrix), len(matrix[0]) if matrix else 0
        matrix = [map(int, row) for row in matrix]
        # check if the first row or colum contains a 1
        max_w = any(chain((matrix[0] if matrix else []), (row[0] for row in matrix)))
        for i in range(1, m):
            for j in range(1, n):
                min_wh = min(matrix[i - 1][j], matrix[i][j - 1])
                is_inc = matrix[i - min_wh][j - min_wh] and matrix[i][j]
                matrix[i][j] = (min_wh if matrix[i][j] else 0) + is_inc
                max_w = max(max_w, matrix[i][j])
        return max_w ** 2

'''109. Convert Sorted List to Binary Search Tree'''
    def sortedListToBST(head):
        lst = []
        cur = head
        while cur:
            lst.append(cur.val)
            cur = cur.next
        def _sortedListToBST(start, end):
            if start >= end:
                return
            mid = (end + start) / 2
            node = TreeNode(lst[mid])
            node.left = _sortedListToBST(start, mid)
            node.right = _sortedListToBST(mid + 1, end)
            return node
        return _sortedListToBST(0, len(lst))

'''36. Valid Sudoku'''
    def isValidSudoku(board):
        def check_section_valid(section):
            from collections import Counter
            counts = Counter(section)
            del counts['.']
            return 1 == max(counts.values() or [1])
        for i in range(9):
            if not check_section_valid(board[i]): return False  # validate row
            if not check_section_valid([board[j][i] for j in range(9)]): return False  # validate col
            x, y = i / 3 * 3, i % 3 * 3
            box = [board[a][b] for a in range(x, x + 3) for b in range(y, y + 3)]
            if not check_section_valid(box): return False  # validate box
        return True

    def isValidSudoku(board):
        hashed = [x for i, row in enumerate(board) for j, el in enumerate(row)
                  if el != '.' for x in [(i/3, j/3, el), (i, el), ('#', j, el)]]
        return len(hashed) == len(set(hashed))

    def isValidSudoku(board):
        visited = set()
        return all(x not in visited and (not visited.add(x))
                   for i, row in enumerate(board) for j, el in enumerate(row)
                   if el != '.' for x in [(i/3, j/3, el), (i, el), ('#', j, el)])

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

'''450. Delete Node in a BST'''
        def _findMinNode(node):
            node = node.right
            while node.left:
                node = node.left
            return node
        def _deleteNode(root, key):
            if not root:
                return
            print root.val
            if root.val > key:
                root.left = _deleteNode(root.left, key)
            elif root.val < key:
                root.right = _deleteNode(root.right, key)
            else:
                if not (root.left and root.right):
                    if root.left:
                        return root.left
                    elif root.right:
                        return root.right
                    else:
                        return
                else:
                    min_node = _findMinNode(root)
                    root.val = min_node.val
                    root.right = _deleteNode(root.right, min_node.val)
            return root

'''113. Path Sum II'''
    def pathSum(root, total):
        cur_path = []
        res = []
        def _pathSum(node, sum_from_root):
            if not node:
                return
            sum_from_root += node.val
            cur_path.append(node.val)
            if sum_from_root == total and not node.left and not node.right:
                res.append(cur_path[:])
            _pathSum(node.left, sum_from_root)
            _pathSum(node.right, sum_from_root)
            cur_path.pop()
        _pathSum(root, 0)
        return res

'''80. Remove Duplicates from Sorted Array II'''
    def removeDuplicates(nums):
        i = 0
        for num in nums:
            if i < 2 or num != nums[i - 2]:
                nums[i] = num
                i += 1
        return i

'''82. Remove Duplicates from Sorted List II'''
    def deleteDuplicates(head):
        prev = dummy = ListNode('dummy')
        cur = dummy.next = head
        while cur and cur.next:
            if cur.val == cur.next.val:
                next_diff = cur
                while next_diff.next and next_diff.val == next_diff.next.val:
                    next_diff = next_diff.next
                prev.next = cur = next_diff.next
            else:
                prev, cur = cur, cur.next
        return dummy.next

'''289. Game of Life'''
    def gameOfLife(board):
        if not board: return
        x_y_diff = [-1, 0, 1]
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                neighbors = [(i + x, j + y) for x in x_y_diff for y in x_y_diff if (x or y)]
                
                count_life = 0
                for x, y in neighbors:
                    if (0 <= x < m) and (0 <= y < n) and (board[x][y] in [1, 2]):
                        count_life += 1

                if board[i][j] and (count_life < 2 or count_life > 3): board[i][j] = 2
                elif not board[i][j] and count_life == 3: board[i][j] = 3

        for i in range(m):
            for j in range(n):
                board[i][j] = 1 & board[i][j]

'''542. 01 Matrix'''
    def updateMatrix(self, matrix):
        def get_adj(i, j):
            return filter(lambda pos: 0 <= pos[0] < len(matrix) and 0 <= pos[1] < len(matrix[0]),
                          [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]])
        bfs = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]:
                    set_none = True
                    for adj_i, adj_j in get_adj(i, j):
                        set_none &= matrix[adj_i][adj_j] == 1 or matrix[adj_i][adj_j] is None
                    if set_none:
                        matrix[i][j] = None
                    else:
                        bfs.append([i, j])
        bfs2 = []
        level = 1
        while bfs:
            level += 1
            while bfs:
                i, j = bfs.pop()
                for adj_i, adj_j in get_adj(i, j):
                    if matrix[adj_i][adj_j] is None:
                        matrix[adj_i][adj_j] = level
                        bfs2.append([adj_i, adj_j])
            bfs, bfs2 = bfs2, bfs
        return matrix

'''24. Swap Nodes in Pairs'''
    def swapPairs(self, head):
        dummy_head = ListNode('dummy')
        dummy_head.next = head
        cur = dummy_head
        while cur and cur.next and cur.next.next:
            nodes = [cur, cur.next, cur.next.next, cur.next.next.next]
            nodes[0].next, nodes[1].next, nodes[2].next, cur = nodes[2], nodes[3], nodes[1], nodes[1]
        return dummy_head.next

'''75. Sort Colors'''
    def sortColors(nums):
        if len(nums) <= 1:
            return
        def _sortColors(start, color):
            i, j = start, len(nums) - 1
            while i < j:
                if nums[j] != color:
                    j -= 1
                elif nums[i] == color:
                    i += 1
                else:
                    nums[i], nums[j] = nums[j], nums[i]
            return i
        
        i = _sortColors(0, 0)
        _sortColors(i + 1 if nums[i] == 0 else i, 1)

'''713. Subarray Product Less Than K'''
    def numSubarrayProductLessThanK(nums, k):
        if k == 0 : return 0
        prod = 1
        start = count = 0
        for end, elem in enumerate(nums):
            prod *= elem
            while prod >= k and start <= end:
                prod /= nums[start]
                start += 1
            count += end - start + 1
        return count

'''49. Group Anagrams'''
    def groupAnagrams(strs):
        group = {}
        for word in strs:
            hashed_bucket = [0] * 26
            for char in word:
                hashed_bucket[ord(char) % len(hashed_bucket)] += 1
            key = hash(str(hashed_bucket))
            group.setdefault(key, [])
            group[key].append(word)
        return group.values()

'''388. Longest Absolute File Path'''
    def lengthLongestPath(input_file_sys):
        input_file_sys = input_file_sys.replace('    ', '\t')
        input_file_sys = input_file_sys.split('\n')
        input_file_sys = map(lambda x: [len(x) - len(x.lstrip('\t')), x.lstrip('\t')], input_file_sys)
        cur_dir = []
        longest_path = ''
        for lvl, file_name in input_file_sys:
            if len(cur_dir) <= lvl:
                cur_dir.append(file_name)
            else:
                cur_dir[lvl] = file_name
                while len(cur_dir) > lvl + 1:
                    cur_dir.pop()
            print '/'.join(cur_dir)
            if '.' in cur_dir[-1] and len(cur_dir) - 1 + sum(map(lambda x: len(x), cur_dir)):
                longest_path = '/'.join(cur_dir)
        return len(longest_path)

'''300. Longest Increasing Subsequence'''
    # dp[i] :=  the index of the last number in sequence of size i
    def lengthOfLIS(self, nums):
        end_idx = [None] * len(nums)
        length = 0
        for i, num in enumerate(nums):
            j = 0
            while j < length and nums[end_idx[j]] < num:
                j += 1
            end_idx[j] = i
            length = max(j + 1, length)
        return length

'''129. Sum Root to Leaf Numbers'''
    def sumNumbers(root):
        def _sumNumbers(node, cur_num=[]):
            if not node:
                return 0
            cur_num.append(node.val)
            if not node.left and not node.right:
                res = int(''.join(map(lambda x: str(x), cur_num)))
            else:
                res =  _sumNumbers(node.left, cur_num) + _sumNumbers(node.right, cur_num)
            cur_num.pop()
            return res
        return _sumNumbers(root)

'''77. Combinations'''
    def combine(n, k):
        cur, res = [], []
        def _combinations(i):
            if len(cur) == k: return res.append(cur[:])
            for j in range(i, n + 1):
                cur.append(j)
                _combinations(j + 1)
                cur.pop()
        _combinations(1)
        return res

'''392. Is Subsequence'''
    def isSubsequence(s, t):
        i = 0
        for char in s:
            while i < len(t) and t[i] != char:
                i += 1
            i += 1
            if i > len(t):
                return False
        return True

'''560. Subarray Sum Equals K'''
    def subarraySum(nums, k):
        sum_from_left = 0
        sum_count = {0:1}
        res = 0
        for i, num in enumerate(nums):
            sum_from_left += num
            target_key = sum_from_left - k
            if target_key in sum_count:
                res += sum_count[target_key]
            sum_count.setdefault(sum_from_left, 0)
            sum_count[sum_from_left] += 1
        return res

'''55. Jump Game'''
    def canJump(nums):
        max_idx_jump = 0
        for i in range(len(nums)):
            if max_idx_jump < i:
                return False
            max_idx_jump = max(max_idx_jump, nums[i] + i)
        return True

'''74. Search a 2D Matrix'''
    def searchMatrix(matrix, target):
        def _searchMatrix(start, end):
            if end <= 0:
                return False
            elif end == start + 1:
                return matrix[start/len(matrix[0])][start%len(matrix[0])] == target
            else:
                mid = (start + end) / 2
                mid_val = matrix[mid/len(matrix[0])][mid%len(matrix[0])]
                if mid_val == target:
                    return True
                elif mid_val < target:
                    return _searchMatrix(mid, end)
                else:
                    return _searchMatrix(start, mid)
        if not matrix:
            return False
        return _searchMatrix(0, len(matrix[0]) * len(matrix))

'''318. Maximum Product of Word Lengths'''
    def maxProduct(words):
        bit_wise_words = []
        for word in words:
            int_word = 0
            for char in word:
                int_word |= 1 << ord(char) % 26
            bit_wise_words.append(int_word)

        max_length = 0
        for i in range(len(words)):
            for j in range(i, len(words)):
                length = len(words[i]) * len(words[j])
                if bit_wise_words[i] & bit_wise_words[j] == 0 and max_length < length:
                    max_length = length
        return max_length

'''227. Basic Calculator II'''
    def calculate(s):
        ops = {
            '+': lambda x, y : x + y,
            '-': lambda x, y : x - y,
            '*': lambda x, y : x * y,
            '/': lambda x, y : x / y
        }
        stack = []
        buf = []
        for char in s.replace(' ', ''):
            if char.isdigit():
                buf.append(char)
            else:
                stack.append(int(''.join(buf)))
                stack.append(char)
                buf = []
        stack.append(int(''.join(buf)))
        new_stack = []
        is_op = False
        for i, item in enumerate(stack):
            if not is_op and new_stack and new_stack[-1] in '*/':
                op, l, r = new_stack.pop(), new_stack.pop(), item
                new_stack.append(ops[op](l, r))
            else:
                new_stack.append(item)
            is_op = not is_op
        new_stack, stack, is_op = [], new_stack, False
        for i, item in enumerate(stack):
            if not is_op and new_stack and new_stack[-1] in '+-':
                op, l, r = new_stack.pop(), new_stack.pop(), item
                new_stack.append(ops[op](l, r))
            else:
                new_stack.append(item)
            is_op = not is_op
        return new_stack[0]

'''114. Flatten Binary Tree to Linked List'''
    def flatten(root):
        def _flatten(node):
            if not node: return
            flatten_left = _flatten(node.left)
            l_end = None
            if flatten_left:
                l_start, l_end = flatten_left
                node.right, l_end.right, node.left = l_start, node.right, None
                flatten_right = _flatten(l_end.right)
            else:
                flatten_right = _flatten(node.right)
            r_start, r_end = flatten_right if flatten_right else [None, None]
            
            if r_end:
                return node, r_end
            elif l_end:
                return node, l_end
            else:
                return node, node
        _flatten(root)

'''378. Kth Smallest Element in a Sorted Matrix'''
    def kthSmallest(matrix, k):
        return sorted(i for row in matrix for i in row)[k-1]

'''395. Longest Substring with At Least K Repeating Characters'''
    def longestSubstring(s, k):
        def _longestSubstring(s, k):
            letter_count = {}
            for char in s:
                letter_count.setdefault(char, 0)
                letter_count[char] += 1
            split_indicies = [-1]
            for i, char in enumerate(s):
                if letter_count[char] < k:
                    split_indicies.append(i)
            split_indicies.append(len(s))
            if len(split_indicies) != 2:
                max_len = 0
                for i in range(1, len(split_indicies)):
                    idx = split_indicies[i]
                    prev_idx = split_indicies[i - 1]
                    max_len = max (max_len, _longestSubstring(s[prev_idx + 1: idx], k))
                return max_len
            else:
                return len(s)
        return _longestSubstring(s, k)

'''382. Linked List Random Node'''
    import random
    class Solution(object):

        def __init__(self, head):
            self.num_elem = 0
            self.head = cur = head
            while cur:
                self.num_elem += 1
                cur = cur.next

        def getRandom(self):
            idx = random.randint(0, self.num_elem - 1)
            cur = self.head
            while cur:
                if idx == 0:
                    return cur.val
                idx -= 1
                cur = cur.next

'''539. Minimum Time Difference'''
    def findMinDifference(self, timePoints):
        timePoints = map(lambda x: [int(i) for i in x.split(':')], timePoints)
        timePoints = map(lambda x: x[0] * 60 + x[1], timePoints)
        min_in_a_day = 24*60
        hash_to_bucket = [False] * min_in_a_day
        for time in timePoints:
            if hash_to_bucket[time]:
                return 0
            hash_to_bucket[time] = True
        prev = None
        first = None
        min_diff = float('inf')
        for i, val in enumerate(hash_to_bucket):
            if val:
                if prev is not None:
                    min_diff = min(i - prev, min_diff)
                else:
                    first = i
                prev = i
        return min(min_in_a_day - (prev - first), min_diff)

'''241. Different Ways to Add Parentheses'''
    import re
    def diffWaysToCompute(input_vals):
        list_sep_vals = []
        buf = []
        list_sep_vals = re.split('([^\d])', input_vals)
        res = []
        ops = { '+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.div }
        for i in range(0, len(list_sep_vals), 2):
            list_sep_vals[i] = int(list_sep_vals[i])
        dp = {}
        def _diffWaysToCompute(start, end):
            print start, end
            if start == end - 1:
                return [list_sep_vals[start]]
            key = '%d_%d'%(start, end)
            if key in dp:
                return dp[key]
            dp[key] = []
            for i in range(start + 1, end, 2):
                left_combo = _diffWaysToCompute(start, i)
                right_combo = _diffWaysToCompute(i + 1, end)
                for l in left_combo:
                    for r in right_combo:
                        dp[key].append(ops[list_sep_vals[i]](l, r))
            return dp[key]
        return _diffWaysToCompute(0, len(list_sep_vals))

'''384. Shuffle an Array'''
    from random import randint
    class Solution(object):
        def __init__(self, nums):
            self.nums = nums
        def reset(self):
            return self.nums
        def shuffle(self):
            self.rand_nums = self.nums[:]
            for i in range(len(self.rand_nums)):
                swap_idx = randint(0, len(self.rand_nums) - 1)
                self.rand_nums[i], self.rand_nums[swap_idx] = self.rand_nums[swap_idx], self.rand_nums[i]
            return self.rand_nums

'''491. Increasing Subsequences'''
    def findSubsequences(nums):
        res = {()}
        for num in nums:
            res |= { ary + (num, ) for ary in res if not ary or ary[-1] <= num }
        return [x for x in res if len(x) >= 2]

'''274. H-Index'''
    def hIndex(citations):
        counter = [0] * (len(citations) + 1)
        for citation in citations:
            if citation > len(citations):
                citation = len(citations)
            counter[citation] += 1
        sum_from_right = 0
        for i in range(len(citations), -1, -1):
            sum_from_right += counter[i]
            if sum_from_right >= i:
                return i

'''678. Valid Parenthesis String'''
    def checkValidString(s):
        high = low = 0
        for i, char in enumerate(s):
            high += -1 if char == ')' else 1
            low = low + 1 if char == '(' else max(low - 1, 0)
            if high < 0:
                return False
        return low == 0

'''454. 4Sum II'''
    def fourSumCount(A, B, C, D):
        from collections import Counter, defaultdict
        sum_count = defaultdict(int, Counter([a + b for a in A for b in B]))
        return sum(sum_count[-d-c] for c in C for d in D)

'''117. Populating Next Right Pointers in Each Node II'''
    def connect(root):
        lvl_head = root
        while lvl_head:
            cur = lvl_head
            next_lvl_head = next_lvl_cur = TreeLinkNode(-1)
            while cur:
                if cur.left:
                    next_lvl_cur.next = cur.left
                    next_lvl_cur = next_lvl_cur.next
                if cur.right:
                    next_lvl_cur.next = cur.right
                    next_lvl_cur = next_lvl_cur.next
                cur = cur.next
            lvl_head = next_lvl_head.next

'''648. Replace Words'''
    def replaceWords(dict, sentence):
        hashed = set()
        for word in dict:
            hashed.add(hash(word))
        res = []
        print hashed
        for word in sentence.split(' '):
            replaced_word = word
            for i in range(1, len(word)):
                if hash(word[:i]) in hashed:
                    replaced_word = word[:i]
                    break
            res.append(replaced_word)
        return ' '.join(res)

'''445. Add Two Numbers II'''
    def addTwoNumbers(self, l1, l2):
        
        def _get_stack(node):
            stack = []
            while node:
                stack.append(node.val)
                node = node.next
            return stack
        s1 = _get_stack(l1)
        s2 = _get_stack(l2)
        
        carry = 0
        dummy = ListNode('dummy')
        while s1 or s2 or carry:
            cur_val = carry
            if s1:
                cur_val += s1.pop()
            if s2:
                cur_val += s2.pop()
            carry, cur_val = cur_val/10, cur_val%10
            cur_node = ListNode(cur_val)
            cur_node.next, dummy.next = dummy.next, cur_node
        return dummy.next

'''153. Find Minimum in Rotated Sorted Array'''
    def findMin(nums):
        low, high = 0, len(nums)
        while low < high:
            mid = (low + high) / 2
            if mid + 1 >= len(nums):
                return min(nums[-1], nums[0])
            elif nums[mid] > nums[mid + 1]:
                return nums[mid + 1]
            elif nums[mid] < nums[0]:
                high = mid
            elif nums[mid] > nums[0]:
                low = mid

'''47. Permutations II'''
    def permuteUnique(nums):
        from collections import Counter
        counts = Counter(nums)
        res = []
        keys = counts.keys()
        cur_perm = []
        def _permuteUnique():
            all_used = True
            for num in keys:
                if num in counts and counts[num]:
                    all_used = False
                    counts[num] -= 1
                    cur_perm.append(num)
                    _permuteUnique()
                    cur_perm.pop()
                    counts[num] += 1
            if all_used:
                res.append(cur_perm[:])
        _permuteUnique()
        return res

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

'''59. Spiral Matrix II'''
    def generateMatrix(n):
        res = [[None] * n for _ in range(n)]
        cur_dir = x = 0
        y = -1
        new_pos_lambda = {
            0: lambda x, y: (x, y + 1),
            1: lambda x, y: (x + 1, y),
            2: lambda x, y: (x, y - 1),
            3: lambda x, y: (x - 1, y)
        }
        for i in range(1, n * n + 1):
            new_x, new_y = new_pos_lambda[cur_dir % 4](x, y)
            if not (0 <= new_x < n and  0 <= new_y < n) or res[new_x][new_y] is not None:
                cur_dir += 1
                new_x, new_y = new_pos_lambda[cur_dir % 4](x, y)
            res[new_x][new_y] = i
            x, y = new_x, new_y
        return res

'''554. Brick Wall'''
    def leastBricks(wall):
        split_count = {}
        split_max = 0
        split_idx = 0
        for row in wall:
            cur_sum = 0
            for i in range(0, len(row) - 1):
                cur_sum += row[i]
                split_count.setdefault(cur_sum, 0)
                split_count[cur_sum] += 1
                split_max = max(split_count[cur_sum], split_max)
        return len(wall) - split_max

'''347. Top K Frequent Elements'''
    def topKFrequent(nums, k):
        from collections import Counter
        counter = Counter(nums)
        freq_to_val = {}
        for x in counter:
            freq_to_val.setdefault(counter[x], [])
            freq_to_val[counter[x]].append(x)
        keys = freq_to_val.keys()
        keys.sort(reverse=True)
        res = []
        for key in keys:
            val = freq_to_val[key]
            while val:
                res.append(val.pop())
                if len(res) == k:
                    return res
        return res

'''368. Largest Divisible Subset'''
    # sort, dp[i] := (size, largest elem) <= max the size by iterating from the beginning if the dp array. given a b c d, if c is divisible by a, and d is divisible by c, then [a c d] will form a subarray since d will be divisible also by a.

'''96. Unique Binary Search Trees'''
    def numTrees(self, n):
        dp = [1, 1]
        if n < len(dp):
            return dp[n]
        for i in range(2, n + 1):
            next_val = 0
            for j in range(1, i + 1):
                next_val += (dp[j - 1]) * (dp[i - j])
            dp.append(next_val)
        return dp[-1]

'''240. Search a 2D Matrix II'''
    def searchMatrix(self, matrix, target):
        if not matrix or not any(matrix):
            return False
        col_i = len(matrix[0]) - 1
        for row in matrix:
            while row[col_i] > target and col_i > 0:
                col_i -= 1
            if row[col_i] == target:
                return True
        return False

'''435. Non-overlapping Intervals'''
    def eraseOverlapIntervals(intervals):
        def not_overlap(int1, int2):
            return min(int1.end, int2.end) <= max(int1.start, int2.start)
        def eft_cmp(x, y):
            if x.end < y.end or x.start < y.start:
                return -1
            elif x.end > y.end or x.start > y.start:
                return 1
            else:
                return 0
        intervals.sort(eft_cmp)
        count = 0
        prev = None
        for interval in intervals:
            if (prev and not_overlap(prev, interval)) or not prev :
                count += 1
                prev = interval
        return len(intervals) - count

'''436. Find Right Interval'''
    def findRightInterval(intervals):
        from bisect import bisect_left
        start_idx = sorted([i.start, idx] for idx, i in enumerate(intervals)) + [[float('inf'), -1]]
        return [start_idx[bisect_left(start_idx, [i.end])][1] for i in intervals]

'''640. Solve the Equation'''
    def solveEquation(equation):
        import re
        def count_x(s):
            return sum(int(num_x)
                       for num_x in re.findall('[\+-]\d*(?=x)',
                                               re.sub('(?<=[\+-])(?=x)', '1', s)))
        def count_num(s): return sum(int(num) for num in re.findall('[\+-]\d+(?=[+\-])', s + '+'))
        left, right = [(x if x.startswith('-') else '+' + x) for x in equation.split('=')]
        num_x = count_x(left) - count_x(right)
        val = count_num(right) - count_num(left)
        if not num_x and not val: return "Infinite solutions"
        elif not num_x: return 'No solution'
        return 'x=%d' % (val / num_x)

'''201. Bitwise AND of Numbers Range'''
    def rangeBitwiseAnd(m, n):
        res = ~0
        while ((m & res) != (n & res)):
            res = res << 1
        return res & m

'''647. Palindromic Substrings'''
    def countSubstrings(s):
        return sum(s[i:j] == s[i:j][::-1] for j in range(len(s) + 1) for i in range(j))

'''120. Triangle'''
    def minimumTotal(triangle):
        for i in range(len(triangle) - 2, -1, -1):
            row = triangle[i]
            next_row = triangle[i + 1]
            for j in range(len(row)):
                row[j] = min(next_row[j], next_row[j + 1]) + row[j]
        return triangle[0][0] if triangle else 0
                

'''515. Find Largest Value in Each Tree Row'''
    def largestValues(root):
        if not root: return []
        bfs = [root]
        res = []
        while bfs:
            res.append(max(map(lambda node: node.val, bfs)))
            bfs = [kid for node in bfs for kid in (node.left, node.right) if kid]
        return res

'''213. House Robber II'''
    def rob(nums):
        def _rob(nums):
            for i in range(1, len(nums)):
                nums[i] = max([nums[i - 1], nums[i]] if i - 2 < 0 else [nums[i - 1], nums[i - 2] + nums[i]])
            return nums[-1] if nums else 0
        return max(_rob(nums[:-1]), _rob(nums[1:])) if len(nums) > 1 else (nums or [0]) [0]

'''535. Encode and Decode TinyURL'''
    class Codec:
        
        def __init__(self):
            self.num_to_url = []
        
        def encode(self, longUrl):
            self.num_to_url.append(longUrl)
            return "http://tinyurl.com/" + str(len(self.num_to_url))

        def decode(self, shortUrl):
            return self.num_to_url[int(shortUrl[int(shortUrl.rfind('/')) + 1:]) - 1]

'''173. Binary Search Tree Iterator'''
    class BSTIterator(object):
        def __init__(self, root):
            cur = root
            self.stack = []
            while cur:
                self.stack.append(cur)
                cur = cur.left

        def hasNext(self):
            return bool(self.stack)

        def next(self):
            cur = ret = self.stack.pop()
            cur = cur.right
            while cur:
                self.stack.append(cur)
                cur = cur.left
            return ret.val

'''338. Counting Bits'''
    def countBits(self, num):
        res = [0]
        for i in range(1, num + 1):
            res.append(res[i >> 1] + (i & 1))
        return res

'''609. Find Duplicate File in System'''
    def findDuplicate(paths):
        from collections import defaultdict
        content_to_path = defaultdict(list)
        for p_f_c in paths:
            p_f_c = p_f_c.split(' ')
            folder = p_f_c[0]
            for i in range(1, len(p_f_c)):
                file_name, content = p_f_c[i][:-1].split('(')
                content_to_path[content].append('%s/%s' %(folder, file_name))
        return [paths for paths in content_to_path.values() if len(paths) > 1]

'''486. Predict the Winner'''
    def PredictTheWinner(nums):
        dp = [[None] * (len(nums) + 1) for _ in range(len(nums) + 1)]
        def _PredictTheWinner(i, j):
            if dp[i][j] is not None:
                return dp[i][j]
            if i == j:
                dp[i][j] = nums[i], 0
            else:
                o_r, s_r = _PredictTheWinner(i, j - 1)
                o_l, s_l = _PredictTheWinner(i + 1, j)
                s_r += nums[j]
                s_l += nums[i]
                if s_r - o_r > s_l - o_l:
                    dp[i][j] =  s_r, o_r
                else:
                    dp[i][j] = s_l, o_l
            return dp[i][j]
        p1, p2 = _PredictTheWinner(0, len(nums) - 1)
        return p1 >= p2
            

'''46. Permutations'''
    def permute(self, nums):
        res = [[]]
        for num in nums:
            new_res =[]
            for item in res:
                for i in range(len(item) + 1):
                    new_res.append(item[:i] + [num] + item[i:])
            res = new_res
        return res

'''725. Split Linked List in Parts'''
    def splitListToParts(root, k):
        cur = root
        count = 0
        while cur:
            count += 1
            cur = cur.next
        remainder, elem_per_part = count % k, count / k
        res = []
        cur = root
        while len(res) != k:
            prev = dummy = ListNode('dummy')
            dummy.next = cur
            for i in range(elem_per_part + (0 if not remainder else 1)):
                prev, cur = cur, cur.next
            prev.next = None
            res.append(dummy.next)
            remainder = max(remainder - 1, 0)
        return res

'''343. Integer Break'''
    def integerBreak(n):
        dp = [None] * (n + 1)
        dp[1] = 1
        if n <= 3: return n - 1
        def get_max(n):
            from itertools import chain
            if dp[n] is not None: return dp[n]
            dp[n] = max(chain((get_max(i) * get_max(n-i) for i in range(1, n)), (n, )))
            return dp[n]
        return get_max(n)

'''22. Generate Parentheses'''
    def generateParenthesis(n):
        def _generateParenthesis(sofar, open_paren, closed_paren, res = []):
            if open_paren == n == closed_paren:
                res.append(sofar)
            elif open_paren <= n:
                if open_paren < n: _generateParenthesis(sofar + '(', open_paren + 1, closed_paren)
                if open_paren > closed_paren: _generateParenthesis(sofar + ')', open_paren, closed_paren + 1)
            return res
        return _generateParenthesis('', 0, 0)

'''89. Gray Code'''
    def grayCode(n):
        gray = [0]
        # take the reverse so that the last of gray is the same as beginning of gray.
        # Loop n times to generate the integers of length n in binary
        for i in range(n):
            gray.extend([g | (1 << i) for g in reversed(gray)])
        return gray

'''623. Add One Row to Tree'''
    def addOneRow(self, root, v, d):
        if d == 1:
            new_node = TreeNode(v)
            new_node.left = root
            return new_node
        bfs = [root]
        prev = None
        for i in range(d - 2):
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        for node in bfs:
            new_node = TreeNode(v)
            node.left, new_node.left = new_node, node.left
            new_node = TreeNode(v)
            node.right, new_node.right = new_node, node.right
        return root

'''78. Subsets'''
    def subsets(nums):
        res = [[]]
        for num in nums:
            res.extend([item + [num] for item in res])
        return res
    def subsets(nums): return [[nums[j] for j in range(len(nums)) if i & (1 << j)] for i in range(1 << len(nums))]

'''73. Set Matrix Zeroes'''
    def setZeroes(matrix):
        m, n = len(matrix), len(matrix[0])
        row_1_zero = not all(matrix[0])
        col_1_zero = not all(row[0] for row in matrix)
        for i, row in enumerate(matrix):
            for j, el in enumerate(row):
                if not el: matrix[i][0] = matrix[0][j] = 0
        for i in range(1, m):
            for j in range(1, n):
                if not matrix[i][0] or not matrix[0][j]: matrix[i][j] = 0
        if row_1_zero: matrix[0] = [0] * n
        if col_1_zero:
            for i in range(m): matrix[i][0] = 0

    def setZeroes(matrix):
        m, n = len(matrix), len(matrix[0])
        # set None indicate row/col contains 0, 0 indicates the elem is 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    for x in range(m):
                        matrix[x][j] = None if matrix[x][j] else matrix[x][j]
                    for y in range(n):
                        matrix[i][y] = None if matrix[i][y] else matrix[i][y]
        for i in range(m):
            for j in range(n):
                matrix[i][j] = matrix[i][j] or 0

'''524. Longest Word in Dictionary through Deleting'''
    def findLongestWord(s, d):
        def subseq_str(s, subseq_s):
            subseq_s = list(subseq_s)[::-1]
            for char in s:
                if char == subseq_s[-1]: subseq_s.pop()
                if not subseq_s: return True
            return False
        d.sort(key=lambda word: (-len(word), word))
        return next((word for word in d if subseq_str(s, word)), "")

'''513. Find Bottom Left Tree Value'''
    def findBottomLeftValue(root):
        prev = []
        bfs = [root]
        while bfs:
            prev = bfs
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return prev[0].val

    def findBottomLeftValue(root):
        def _findBottomLeftValue(node, depth=0, res=[]):
            if not node: return res
            if depth >= len(res):
                res.append(node.val)
            _findBottomLeftValue(node.left, depth + 1, res)
            _findBottomLeftValue(node.right, depth + 1, res)
            return res
        return _findBottomLeftValue(root)[-1]

'''654. Maximum Binary Tree'''
    def constructMaximumBinaryTree(nums):
        def _constructMaximumBinaryTree(i, j):
            if i == j:
                return
            max_idx, max_val = -1, float('-inf')
            for k in range(i, j):
                if max_val < nums[k]:
                    max_idx, max_val = k, nums[k]
            node = TreeNode(nums[max_idx])
            node.left = _constructMaximumBinaryTree(i, max_idx)
            node.right = _constructMaximumBinaryTree(max_idx + 1, j)
            return node
        return _constructMaximumBinaryTree(0, len(nums))
    def constructMaximumBinaryTree(nums):
        stack = []
        for num in nums:
            node = TreeNode(num)
            while stack and stack[-1].val < num:
                node.left = stack.pop()
            if stack:
                stack[-1].right = node
            stack.append(node)
        return stack[0]

'''655. Print Binary Tree'''
    def printTree(self, root):
        res = []
        bfs = [root]
        while any(bfs):
            res.append(bfs)
            bfs = [(kid if kid else '') \
                   for node in bfs for kid in (['', ''] if not node else [node.left, node.right])]
        padding = 0
        def convert_to_str(node): return str(node.val) if type(node) != str else ''
        def add_spaces(lst, num_spaces):
            for i in range(num_spaces): lst.append('')
        for i in range(len(res) - 1, -1, -1):
            spacing = padding * 2 + 1
            new_row = [""] * padding
            for j in range(len(res[i]) - 1):
                new_row.append(convert_to_str(res[i][j]))
                add_spaces(new_row, spacing)
            new_row.append(convert_to_str(res[i][-1]))
            add_spaces(new_row, padding)
            res[i] = new_row
            padding = spacing
        return res

'''419. Battleships in a Board'''
    def countBattleships(board):
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                if i + 1 < m and 'X' == board[i + 1][j] or \
                    j + 1 < n and 'X' == board[i][j + 1]:
                    board[i][j] = '.'
        return sum(1 for row in board for el in row if el == 'X')

'''529. Minesweeper'''
    def updateBoard(board, click):
        def get_adjs(i, j):
            dif = [-1, 0, 1]
            xy_dir = [(x, y) for x in dif for y in dif if x or y]
            adjs = [(i + x, j + y) for x, y in xy_dir]
            adjs = [(x, y) for x, y in adjs if 0 <= x < len(board) and 0 <= y < len(board[0])]
            return adjs
        def click_board(i, j):
            if board[i][j] == 'M':
                board[i][j] = 'X'
            elif board[i][j] == 'E':
                adjs = get_adjs(i, j)
                board[i][j] = str(sum(1 for x, y in adjs if board[x][y] == 'M') or 'B')
                if board[i][j] == 'B':
                    for x, y in adjs: click_board(x, y)
        click_board(*click)
        return board

'''31. Next Permutation'''
    def nextPermutation(self, nums):
        if len(nums) <= 1: return
        l, r = 0, len(nums) - 1
        # find first decreasing pair from the right,
        # then swap it with the smallest element that's strictly greater.
        for i in xrange(len(nums) - 2, -1, -1):
            if nums[i] < nums[i+1]:
                right_greater = min(((nums[j], j) for j in range(i + 1, len(nums)) if nums[j] > nums[i]), \
                                    key=lambda x: (x[0], -x[1]))[1]
                nums[i], nums[right_greater] = nums[right_greater], nums[i]
                l = i + 1
                break
        # reverse the elems to the right
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l, r = l + 1, r - 1

'''508. Most Frequent Subtree Sum'''
    def findFrequentTreeSum(root):
        if not root: return []
        from collections import defaultdict
        val_to_freq = defaultdict(int)
        def _findFrequentTreeSum(node):
            if not node: return 0
            tree_sum = node.val + _findFrequentTreeSum(node.left) + _findFrequentTreeSum(node.right)
            val_to_freq[tree_sum] += 1
            return tree_sum
        _findFrequentTreeSum(root)
        max_freq = max(val_to_freq.values())
        return [val for val, freq in val_to_freq.iteritems() if max_freq == freq]

'''260. Single Number III'''
    def singleNumber(nums):
        xord = 0
        for num in nums: xord ^= num

        for i in range(32):
            if (1 << i) & xord:
                num1 = 0
                num2 = 0
                for num in nums:
                    if (1 << i) & num:
                        num1 ^= num
                    else:
                        num2 ^= num
                return [num1, num2]

'''495. Teemo Attacking'''
    def findPoisonedDuration(timeSeries, duration):
        total_time = 0
        for i, time in enumerate(timeSeries):
            total_time += duration
            time_diff = timeSeries[i] - timeSeries[i-1]
            if i > 0 and time_diff < duration:
                total_time -= duration - time_diff
        return total_time

'''677. Map Sum Pairs'''
    class MapSum(object):

        def __init__(self):
            self.trie_tree = {}

        def insert(self, key, val):
            cur = self.trie_tree
            for char in key:
                cur.setdefault(char, {})
                cur = cur[char]
            cur['val'] = val

        def sum(self, prefix):
            res = 0
            cur = self.trie_tree
            for char in prefix:
                if char not in cur:
                    return 0
                cur = cur[char]
            bfs = [cur]
            vals = []
            while bfs:
                vals.extend([item for item in bfs if type(item) == int])
                bfs = [node[char] for node in bfs for char in (node if type(node) != int else [])]
            return sum(vals)

'''394. Decode String'''
    def decodeString(s):
        stack = []
        str_queue = []
        def add_to_queue(str_queue, stack):
            stack.append(''.join(str_queue))
            del str_queue[:]
            last = []
            while stack and not stack[-1].isdigit(): last.append(stack.pop())
            if last: stack.append(''.join(list(reversed(last))))
        for char in s:
            if '[' == char:
                add_to_queue(str_queue, stack)
            elif ']' == char:
                add_to_queue(str_queue, stack)
                stack.append(stack.pop() * int(stack.pop()))
            elif str_queue and str_queue[-1].isdigit() != char.isdigit():
                add_to_queue(str_queue, stack)
                str_queue.append(char)
            else: str_queue.append(char)
        stack.append(''.join(str_queue))
        return ''.join(stack)

'''376. Wiggle Subsequence'''
    def wiggleMaxLength(nums):
        # update up_tog when the first up/down, or down/up is seen.
        # if up_tog is set, and at the current position, it goes in the
        # same direction of up_tog, then we toggle the direction and 
        # add to the count.
        if not nums: return 0
        count = 1
        up_tog = None
        for i in range(1, len(nums)):
            if up_tog is None and nums[i-1] != nums[i]:
                up_tog = nums[i-1] < nums[i]
            if (up_tog is not None) and \
             (nums[i-1] < nums[i] and up_tog or \
              nums[i-1] > nums[i] and not up_tog):
                count, up_tog = count + 1, not up_tog
        return count

'''236. Lowest Common Ancestor of a Binary Tree'''
    def lowestCommonAncestor(self, root, p, q):
        def _searchAncestor(node):
            if node in [p, q, None]: return node
            else:
                l, r = _searchAncestor(node.left), _searchAncestor(node.right)
                return node if (l and r) else (l or r)
        return _searchAncestor(root)

'''199. Binary Tree Right Side View'''
    def rightSideView(root):
        if not root: return []
        bfs = [root]
        res = []
        while bfs:
            res.append(bfs[-1].val)
            bfs = [kid for node in bfs for kid in (node.left, node.right) if kid]
        return res

'''398. Random Pick Index'''
    class Solution(object):
        def __init__(self, nums): self.nums = nums

        def pick(self, target):
            import random
            return random.choice([i for i, num in enumerate(self.nums) if num == target])

'''652. Find Duplicate Subtrees'''
    def findDuplicateSubtrees(root):
        def _findDuplicateSubtrees(node, hash_to_count_node):
            if not node: return
            l_res = _findDuplicateSubtrees(node.left, hash_to_count_node)
            r_res = _findDuplicateSubtrees(node.right, hash_to_count_node)
            serial = (node.val, l_res, r_res)
            hash_val = hash(str(serial))
            hash_to_count_node.setdefault(hash_val, [0, node])
            hash_to_count_node[hash_val][0] += 1
            return serial

        hash_to_count_node = {}
        _findDuplicateSubtrees(root, hash_to_count_node)
        return [node for i, node in hash_to_count_node.values() if i >= 2]

'''92. Reverse Linked List II'''
    def reverseBetween(head, m, n):
        dummy = cur = ListNode(0)
        cur.next = head
        for i in range(m): prev, cur = cur, cur.next
        tail1, tail2 = prev, cur
        prev = None
        for i in range(n - m + 1):
            cur.next, prev, cur = prev, cur, cur.next
        tail1.next, tail2.next = prev, cur
        return dummy.next

'''90. Subsets II'''
    def subsetsWithDup(nums):
        from collections import Counter
        counts = Counter(nums)
        uniq = counts.keys()
        cur, res = [], [[]]
        def _subsetsWithDup(idx=0):
            if idx >= len(uniq): return
            if counts[uniq[idx]]:
                counts[uniq[idx]] -= 1
                cur.append(uniq[idx])
                res.append(cur[:])
                _subsetsWithDup(idx)
                cur.pop()
                counts[uniq[idx]] += 1
            _subsetsWithDup(idx + 1)
        _subsetsWithDup()
        return res

'''230. Kth Smallest Element in a BST'''
    def kthSmallest(root, k):
        stack = []
        def move_left(node):
            while node:
                stack.append(node)
                node = node.left
        move_left(root)
        for i in range(k):
            next_elem = stack.pop()
            if next_elem.right:
                move_left(next_elem.right)
        return next_elem.val

'''525. Contiguous Array'''
    def findMaxLength(nums):
        for i in range(len(nums)): nums[i] = -1 if nums[i] == 0 else 1
        sum_to_idx, sum_so_far, max_len = {0: -1}, 0, 0
        for i in range(len(nums)):
            sum_so_far += nums[i]
            if sum_so_far in sum_to_idx:
                max_len = max(i - sum_to_idx[sum_so_far], max_len)
            sum_to_idx.setdefault(sum_so_far, i)
        return max_len

'''64. Minimum Path Sum'''
    def minPathSum(grid):
        for i in range(1, len(grid)): grid[i][0] += grid[i - 1][0]
        for j in range(1, len(grid[0])): grid[0][j] += grid[0][j - 1]
        for i in range(1, len(grid)):
            for j in range(1, len(grid[0])):
                grid[i][j] += min(grid[i-1][j], grid[i][j - 1])
        return grid[-1][-1]

'''19. Remove Nth Node From End of List'''
    def removeNthFromEnd(head, n):
        nth = head
        for _ in range(n): nth = nth.next
        nth, cur, prev = head, nth, None
        while cur: cur, prev, nth = cur.next, nth, nth.next
        if nth == head: return head.next
        if prev and prev.next: prev.next = prev.next.next
        return head

'''729. My Calendar I'''
    class MyCalendar(object):

        def __init__(self):
            self.intervals = []

        def book(self, start, end):
            for int_start, int_end in self.intervals:
                if not (int_end <= start or end <= int_start):
                    return False
            self.intervals.append((start,end))
            return True

'''187. Repeated DNA Sequences'''
    def findRepeatedDnaSequences(s):
        visited = set()
        res = []
        for i in range(0, len(s) - 10 + 1):
            sub_s = s[i: i + 10]
            if sub_s in visited and sub_s not in res:
                res.append(sub_s)
            visited.add(sub_s)
        return res

'''611. Valid Triangle Number'''
    def triangleNumber(nums):
        res = 0
        nums.sort()
        res = 0
        for i in range(len(nums) -1, 1, -1):
            l, r = 0, i - 1
            while l < r:
                if nums[l] + nums[r] > nums[i]:
                    res += (r - l)
                    r -= 1
                else:
                    l += 1
        return res

'''462. Minimum Moves to Equal Array Elements II'''
    def minMoves2(nums):
        nums.sort()
        mid_val = nums[len(nums) / 2]
        return sum(abs(x - mid_val) for x in nums)

'''452. Minimum Number of Arrows to Burst Balloons'''
    def findMinArrowShots(points):
        points.sort()
        start = None
        res = 0
        while points:
            last = points.pop()
            if start is None or not (last[0] <= start <= last[1]):
                start = last[0]
                res += 1
        return res

'''494. Target Sum'''
    def findTargetSumWays(nums, S):
        dp = {0: 1}
        for num in nums:
            new_dp = {}
            for key, val in dp.iteritems():
                for new_key in [key + num, key - num]:
                    new_dp.setdefault(new_key, 0)
                    new_dp[new_key] += val
            dp = new_dp
        return dp.get(S, 0)

'''547. Friend Circles'''
    def findCircleNum(M):
        visited = set()
        res = 0
        for node in range(len(M)):
            if node not in visited:
                res += 1
                bfs = [node]
                while bfs:
                    visited |= set(bfs)
                    bfs = [adj for i in bfs for adj, val in enumerate(M[i]) if val and (adj not in visited)]
        return res

'''423. Reconstruct Original Digits from English'''
    def originalDigits(self, s):
        '''
        zero: Only digit with z
        two: Only digit with w
        four: Only digit with u
        six: Only digit with x
        eight: Only digit with g
        '''
        from collections import Counter
        s = Counter(s)
        idx_to_word = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        def set_count(char, num, counts):
            counts[num] = s.get(char, 0) - sum(counts[i] for i, word in enumerate(idx_to_word) if char in word and i != num)
        counts = [
            s.get('z', 0), None,
            s.get('w', 0), None,
            s.get('u', 0), None,
            s.get('x', 0), None,
            s.get('g', 0), None,
        ]
        set_count('o', 1, counts)
        set_count('t', 3, counts)
        set_count('f', 5, counts)
        set_count('s', 7, counts)
        set_count('i', 9, counts)
        return ''.join(str(i) * occ for i, occ in enumerate(counts))
            
            583. Delete Operation for Two Strings
    def minDistance(word1, word2):
        if not word1 or not word2: return len(word1) or len(word2)
        dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
        for i in range(len(word1)):
            for j in range(len(word2)):
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j], dp[i][j] + (1 if word1[i] == word2[j] else 0))
        return (len(word1) + len(word2) - dp[-1][-1] * 2)

'''40. Combination Sum II'''
    def combinationSum2(candidates, target):
        from collections import Counter
        counts = Counter(candidates)
        uniq_nums = counts.keys()
        cur, res = [], []
        def _combinationSum2(target, idx=0):
            if idx >= len(uniq_nums) or target < 0: return
            if target == 0: res.append(cur[:])
            else:
                if counts[uniq_nums[idx]]:
                    counts[uniq_nums[idx]] -= 1
                    cur.append(uniq_nums[idx])
                    _combinationSum2(target - uniq_nums[idx] , idx)
                    cur.pop()
                    counts[uniq_nums[idx]] += 1
                _combinationSum2(target, idx + 1)
        _combinationSum2(target)
        return res

'''658. Find K Closest Elements'''
    def findClosestElements(arr, k, x):
        from bisect import bisect_left
        pos = bisect_left(arr, x, 0, len(arr))
        if pos - 1 > 0 and arr[pos] != x: pos -= 1
        i, j = pos, pos + 1
        while j - i < k:
            if 0 < i and j < len(arr):
                if ((x - arr[i - 1]) <= (arr[j] - x)): i -= 1
                else: j += 1
            elif 0 < i: i -= 1
            else: j += 1
        return arr[i:j]

'''139. Word Break'''
    def wordBreak(self, s, wordDict):
        dp = [False] * (len(s) + 1)
        dp[0] = True
        wordDict = set(wordDict)
        for j in range(1, len(s) + 1):
            for i in range(j - 1, -1, -1):
                if dp[i] and s[i: j] in wordDict:
                    dp[j] = True
                    continue
        return dp[-1]

'''142. Linked List Cycle II'''
    def detectCycle(head):
        if not head or not head.next: return
        slow, fast = head.next, head.next.next
        while (fast and fast.next) and slow != fast:
            slow = slow.next
            fast = fast.next.next
        if not fast or not fast.next: return
        cur = head
        in_loop_cur = fast
        while cur != in_loop_cur:
            cur = cur.next
            in_loop_cur = in_loop_cur.next
        return cur

'''373. Find K Pairs with Smallest Sums'''
    def kSmallestPairs(nums1, nums2, k):
        return [item[1:] for item in sorted([(sum([num1, num2]), num1, num2) for num1 in nums1[:k] for num2 in nums2[:k]])[:k]]

'''576. Out of Boundary Paths'''
    def findPaths(m, n, N, i, j):
        diff = zip([-1, 0, 1, 0],
                   [0, -1, 0, 1])
        def move(bfs):
            from collections import defaultdict
            new_bfs = defaultdict(int)
            for (x, y, step), count in bfs.iteritems():
                for x_dif, y_dif in diff:
                    if 0 <= (x + x_dif) < m and 0 <= (y + y_dif) < n:
                        new_bfs[(x + x_dif, y + y_dif, step + 1)] += count
            return  new_bfs, sum(new_bfs.values())

        bfs = {(i, j, 0): 1}
        total = 0
        pre_count = 1
        while bfs and N > 0:
            bfs, count = move(bfs)
            total += pre_count * 4 - count
            pre_count = count
            N -= 1
        return total % (10**9 + 7)

    def findPaths(m, n, N, i, j):
        grid = [[0] * n for _ in range(m)]
        dif = zip([1, 0, -1, 0],
                  [0, 1, 0, -1])
        grid[i][j] = 1
        def sum_neighbor():
            new_grid = [[0] * n for _ in range(m)]
            for x in range(m):
                for y in range(n):
                    adjs = [(x + x_dir, y + y_dir) for x_dir, y_dir in dif]
                    new_grid[x][y] += sum(grid[a][b] for a, b in adjs if 0 <= a < m and 0 <= b < n)
            return new_grid
        prev_count = 1
        total_count = 0
        for _ in range(N):
            grid = sum_neighbor()
            cur_count = sum(map(sum, grid))
            total_count += prev_count * 4 - cur_count
            prev_count = cur_count
        return total_count %(10**9 + 7)

    def findPaths(m, n, N, i, j):
        grid = [[0] * n for _ in range(m)]
        dif = zip([0, 1, 0, -1],
                  [1, 0, -1, 0])
        for _ in range(N):
            grid = [[
                sum(grid[adj_x][adj_y] if (0 <= adj_x < m and 0 <= adj_y < n) else 1
                    for adj_x, adj_y in [(x + x_dir, y + y_dir) for x_dir, y_dir in dif])
                for y in range(n)]
                for x in range(m)]
        return grid[i][j] % (10**9 + 7)

'''200. Number of Islands'''
    def numIslands(grid):
        if not grid or not grid[0]: return 0
        def convert(from_sym, to_sym, i, j):
            def get_adj(i, j):
                return [(x, y) for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                        if 0 <= x < len(grid) and 0 <= y < len(grid[0])]
            if grid[i][j] == from_sym:
                grid[i][j] = to_sym
                for adj in get_adj(i, j): convert(from_sym, to_sym, adj[0], adj[1])
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    res += 1
                    convert('1', None, i, j)
        return res

'''94. Binary Tree Inorder Traversal'''
    def inorderTraversal(root):
        stack = []
        def move_left(node):
            while node:
                stack.append(node)
                node = node.left
        move_left(root)
        res = []
        while stack:
            next_elem = stack.pop()
            res.append(next_elem.val)
            if next_elem.right: move_left(next_elem.right)
        return res

'''116. Populating Next Right Pointers in Each Node'''
    def connect(root):
        if not root:
            return
        lvl = root
        while lvl.left:
            cur = lvl
            while cur:
                cur.left.next = cur.right
                if cur.next:
                    cur.right.next = cur.next.left
                cur = cur.next
            lvl = lvl.left

'''662. Maximum Width of Binary Tree'''
    def widthOfBinaryTree(root):
        if not root: return
        bfs = [(root, 1)]
        max_len = 0
        while bfs:
            max_len = max(max_len, bfs[-1][1] - bfs[0][1] + 1)
            bfs = [(kid, pos * 2 + (kid == node.right))
                   for node, pos in bfs
                   for kid in (node.left, node.right) if kid]
        return max_len

'''670. Maximum Swap'''
    def maximumSwap(num):
        num = [int(i) for i in list(str(num))]
        max_from_right = [None] * len(num)
        for i in range(len(num) - 1, -1, -1):
            if i == len(num) - 1 or num[max_from_right[i + 1]] < num[i]:
                max_from_right[i] = i
            else: max_from_right[i] = max_from_right[i + 1]
        for i in range(len(num)):
            if max_from_right[i] != i and num[i] != num[max_from_right[i]] :
                num[i], num[max_from_right[i]] = num[max_from_right[i]], num[i]
                break
        return int(''.join([str(item) for item in num]))

'''162. Find Peak Element'''
    def findPeakElement(nums):
        if not nums: return 0
        def get_val(idx):
            return nums[idx] if 0 <= idx < len(nums) else float('-inf')
        lo, hi = 0, len(nums) - 1
        while True:
            mid = (lo + hi) / 2
            if get_val(mid - 1) < get_val(mid) > get_val(mid + 1): return mid
            elif get_val(mid - 1) > get_val(mid): hi = mid - 1
            else: lo = mid + 1

'''223. Rectangle Area'''
    def computeArea(A, B, C, D, E, F, G, H):
        width, height = (min(C, G) - max(A, E)), (min(D, H) - max(B, F))
        overlap = width * height if width > 0 and height > 0 else 0
        area1 = (C - A) * (D - B)
        area2 = (G - E) * (H - F)
        return area1 + area2 - overlap

'''556. Next Greater Element III'''
    def nextGreaterElement(n):
        num = list(str(n))
        n = len(num)
        for i in range(n - 2, -1, -1):
            if num[i] < num[i + 1]:
                next_largest = i + 1
                for j in range(i + 2, n):
                    if num[i] < num[j] <= num[next_largest]:
                        next_largest = j
                num[i], num[next_largest] = num[next_largest], num[i]
                next_greater = int(''.join(num[:i + 1] + num[i + 1:][::-1]))
                return next_greater if next_greater < (1<<31) else -1
        return -1

'''449. Serialize and Deserialize bst'''
    class Codec:
        def serialize(self, root):
            def _serialize(node):
                if not node: return
                return (node.val, _serialize(node.left), _serialize(node.right))
            return str(_serialize(root))

        def deserialize(self, data):
            def _deserialize(input_tuple):
                if not input_tuple: return
                node = TreeNode(input_tuple[0])
                node.left = _deserialize(input_tuple[1])
                node.right = _deserialize(input_tuple[2])
                return node
            return  _deserialize(eval(data))

'''698. Partition to K Equal Sum Subsets'''
    def canPartitionKSubsets(nums, k):
        if len(nums) < k: return False
        total = sum(nums)
        if total % k != 0: return False
        nums.sort()
        self.max_val = total / k
        def _canPartitionKSubsets(buckets):
            if not nums: return buckets.count(self.max_val) == len(buckets)
            next_elem = nums.pop()
            visited_bucket = set()
            for i, bucket in enumerate(buckets):
                if buckets[i] not in visited_bucket and buckets[i] + next_elem <= self.max_val:
                    buckets[i] += next_elem
                    if _canPartitionKSubsets(buckets): return True
                    buckets[i] -= next_elem
                    visited_bucket.add(buckets[i])
            nums.append(next_elem)
            return False
        return _canPartitionKSubsets([0] * k)

'''332. Reconstruct Itinerary'''
    def findItinerary(tickets):
        from_to = collections.defaultdict(list)
        for dept, arr in tickets: from_to[dept].append(arr)
        for dept, arrs in from_to.iteritems(): arrs.sort(reverse=True)
        def dfs(airport, route=[]):
            while from_to[airport]:
                dfs(from_to[airport].pop(), route)
            route.append(airport)
            return route
        return dfs('JFK')[::-1]

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

'''93. Restore IP Addresses'''
    def restoreIpAddresses(s):
        n = len(s)
        if float(n) / 4 > 3: return []
        dp = [[] for _ in range(n + 1)]
        dp[0].append([])
        for j in range(1, n + 1):
            for i in range(max(j - 3, 0), j):
                for ip in dp[i]:
                    if ((s[i:j] == '0') or \
                        (not s[i:j].startswith('0') and int(s[i:j]) < 256)) \
                         and len(ip) < 4:
                        dp[j].append(ip + [s[i:j]])

        return ['.'.join(ip) for ip in dp[n] if len(ip) == 4]

'''95. Unique Binary Search Trees II'''
    def generateTrees(n):
        lst = range(1, n + 1)
        if not n: return []
        def clone(node):
            if not node: return
            new_node = TreeNode(node.val)
            new_node.left = clone(node.left)
            new_node.right = clone(node.right)
            return new_node
        def _generateTrees(i, j):
            if i >= j: return [None]
            res = []
            for k in range(i, j):
                left = _generateTrees(i, k)
                right = _generateTrees(k + 1, j)
                for l in left:
                    for r in right:
                        node = TreeNode(lst[k])
                        node.left = clone(l)
                        node.right = clone(r)
                        res.append(node)
            return res
        return _generateTrees(0, len(lst))

'''692. Top K Frequent Words'''
    def topKFrequent(words, k):
        word_to_freq = {}
        for word in words:
            word_to_freq.setdefault(word, 0)
            word_to_freq[word] += 1
        def cmp_func(freq_word1, freq_word2):
            if freq_word1[0] > freq_word2[0] or freq_word1[0] == freq_word2[0] and freq_word1[1] < freq_word2[1]:
                return -1
            return 1
        freq = [(freq, word) for word, freq in word_to_freq.iteritems()]
        freq.sort(cmp_func)
        return [item[1] for item in freq[:k]]

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

'''148. Sort List'''
    def sortList(head):
        def _merge_sort(head):
            def _get_from_queue(h1, h2):
                if not h1 and not h2: return None, None, None
                elif not h1 or (h1 and h2 and h2.val < h1.val): return h2, h1, h2.next
                elif not h2 or (h1 and h2 and h2.val >= h1.val): return h1, h1.next, h2
            if not head or not head.next: return head
            slow = fast = head
            while fast and fast.next and fast.next.next: slow, fast = slow.next, fast.next.next
            h1, h2, slow.next = head, slow.next, None
            h1 = _merge_sort(h1)
            h2 = _merge_sort(h2)
            cur = dummy = ListNode('dummy')
            while cur:
                cur.next, h1, h2 = _get_from_queue(h1, h2)
                cur = cur.next
            return dummy.next
        return _merge_sort(head)

'''147. Insertion Sort List'''
    def insertionSortList(head):
        dummy = ListNode(-1000000)
        dummy.next = head
        boundary_prev, boundary = dummy, head
        while boundary:
            node = boundary
            boundary = boundary.next
            boundary_prev.next = boundary
            prev, cur, node.next = dummy, dummy.next, None
            while cur and cur != boundary and cur.val < node.val: prev, cur = cur, cur.next
            tmp = prev.next
            prev.next = node
            node.next = tmp
            if boundary_prev.next != boundary: boundary_prev = boundary_prev.next
        return dummy.next

'''456. 132 Pattern'''
    def find132pattern(nums):
        stack = []
        s3 = float('-inf')
        for num in reversed(nums):
            if num < s3: return True
            while stack and stack[-1] < num: s3 = stack.pop()
            stack.append(num)
        return False

'''406. Queue Reconstruction by Height'''
    def reconstructQueue(people):
        people.sort(key=lambda(h, k): (-h, k))
        res = []
        for p in people:
            res.insert(p[1], p)
        return res

'''442. Find All Duplicates in an Array'''
    def findDuplicates(nums):
        res = []
        for i in range(len(nums)):
            idx = abs(nums[i]) - 1
            if nums[idx] < 0: res.append(idx + 1)
            else: nums[idx] = -nums[idx]
        return res
        

'''540. Single Element in a Sorted Array'''
    def singleNonDuplicate(nums):
        res = 0
        for num in nums: res ^= num
        return res

'''131. Palindrome Partitioning'''
    def partition(s):
        n = len(s) + 1
        dp = [[[]]]
        for j in range(1, n):
            dp.append(
                [prefix + [s[i:j]]
                 for i in range(j)
                 if s[i:j] == s[i:j][::-1]
                 for prefix in dp[i]])
        return dp[-1]

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

'''451. Sort Characters By Frequency'''
    def frequencySort(s):
        from collections import Counter
        counts = Counter(s)
        counts = [(freq, char) for char, freq in counts.iteritems()]
        counts.sort(reverse=True)
        for i in range(len(counts)): counts[i] = counts[i][1] * counts[i][0]
        return ''.join(counts)

'''228. Summary Ranges'''
    def summaryRanges(nums):
        ranges = []
        for i in range(len(nums)):
            if not ranges or ranges[-1][1] != nums[i] - 1:
                ranges.append([nums[i], nums[i]])
            else:
                ranges[-1][1] = nums[i]
        return [str(i) + '->' + str(j) if i != j else str(i) for i, j in ranges]

'''337. House Robber III'''
    def rob(root):
        def _rob(node):
            if not node: return 0, 0
            inc_l, not_inc_l = _rob(node.left)
            inc_r, not_inc_r = _rob(node.right)
            inc_node = node.val + not_inc_l + not_inc_r
            not_inc_node = max(inc_l, not_inc_l) + max(inc_r, not_inc_r)
            return inc_node, not_inc_node
        return max(_rob(root))

'''319. Bulb Switcher'''
    def bulbSwitch(n):
        return int(n ** (1.0/2))

'''424. Longest Repeating Character Replacement'''
    def characterReplacement(s, k):
        counts = {}
        max_len = start_i = 0
        for end_i, end in enumerate(s):
            counts[end] = counts.get(end, 0) + 1
            max_len = max(max_len, counts[end])
            while ((end_i - start_i + 1) - max_len) > k:
                counts[s[start_i]] -= 1
                start_i += 1
        return max_len + min(k, len(s) - max_len)

'''328. Odd Even Linked List'''
    def oddEvenList(head):
        even_odd_head = [ListNode("even"), ListNode("odd")]
        even_odd = even_odd_head[:]
        cur = head
        toggle = 0
        while cur:
            even_odd[toggle].next = ListNode(cur.val)
            even_odd[toggle] = even_odd[toggle].next
            toggle = 1 - toggle
            cur = cur.next
        even_odd[0].next = even_odd_head[1].next
        return even_odd_head[0].next

'''2. Add Two Numbers'''
    def addTwoNumbers(l1, l2):
        l1_cur, l2_cur = l1, l2
        cur = res = ListNode('dummy')
        carry = 0
        while l1 or l2 or carry:
            digit_sum = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry
            carry, digit = divmod(digit_sum, 10)
            cur.next = ListNode(digit)
            cur = cur.next
            if l1: l1 = l1.next
            if l2: l2 = l2.next
        return res.next

'''60. Permutation Sequence'''
    def nextPermutation(nums):
        if len(nums) <= 1: return
        # find first decreasing seq
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] < nums[i+1]: break
        j, j_val = None, float('inf')
        # find strictly 1 greater
        for k in range(i + 1, len(nums)):
            if nums[i] < nums[k] <= j_val:
                j, j_val = k, nums[k]
        if j is not None:
            # swap i, j
            nums[i], nums[j] = nums[j], nums[i]
        else: i = -1
        # reverse remaining
        i, j = i + 1, len(nums) - 1
        while i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1

'''144. Binary Tree Preorder Traversal'''
    def preorderTraversal(root):
        if not root: return []
        bfs = [root]
        def check_node(node):
            if type(node) == TreeNode: return [node.val, node.left, node.right]
            else: return [node]
        while any(type(node) == TreeNode for node in bfs):
            bfs = [kid for node in bfs for kid in check_node(node) if kid is not None]
        return bfs

'''11. Container With Most Water'''
    def maxArea(height):
        i, j = 0, len(height) - 1
        water = 0
        while i < j:
            h = min(height[i], height[j])
            water = max(h * (j - i), water)
            while i < j and height[j] <= h: j -= 1
            while i < j and height[i] <= h: i += 1
        return water

'''334. Increasing Triplet Subsequence'''
    def increasingTriplet(nums):
        first = second = float('inf')
        for n in nums:
            if n <= first: first = n
            elif n <= second: second = n
            else: return True
        return False

    def increasingTriplet(nums):
        max_last_two = []
        cur_max = float('-inf')
        for num in reversed(nums):
            if cur_max > num: return True
            elif not max_last_two or num > max_last_two[0]: max_last_two = [num]
            elif len(max_last_two) == 2 and max_last_two[0] > num > max_last_two[1]: max_last_two[1] = num
            elif len(max_last_two) == 1 and max_last_two[0] > num: max_last_two.append(num)
            if len(max_last_two) == 2: cur_max = max(cur_max, max_last_two[1])
        return False

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

'''416. Partition Equal Subset Sum'''
    def canPartition(nums):
        all_sums = {0}
        for num in nums: all_sums |= set(part_total + num for part_total in all_sums)
        return (float(sum(nums)) / 2) in all_sums

'''16. 3Sum Closest'''
    def threeSumClosest(nums, target):
        min_diff = float('inf')
        min_num = None
        nums.sort()
        for i in range(0, len(nums) - 2):
            j, k = i + 1, len(nums) - 1
            while j < k:
                i_j_k_sum = nums[i] + nums[j] + nums[k]
                diff = abs(i_j_k_sum - target)
                if min_diff > diff:
                    min_diff = diff
                    min_sum = i_j_k_sum
                if i_j_k_sum > target: k -= 1
                else: j += 1
        return min_sum

'''81. Search in Rotated Sorted Array II'''
    def search(nums, target):
        if not nums: return False
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            while (hi - lo) >= 1 and nums[hi] == nums[hi - 1]: hi -= 1
            mid = (lo + hi) / 2
            if nums[mid] == target: return True
            elif nums[lo] <= target < nums[mid] or \
                (nums[lo] > nums[mid] and (target < nums[mid] or target >= nums[lo])): hi = mid - 1
            else: lo = mid + 1
        return False

'''48. Rotate Image'''
    def rotate(matrix):
        for i in range(len(matrix) / 2):
            for j in range(i, len(matrix) - i - 1):
                n = len(matrix) - 1
                vals = matrix[i][j], matrix[j][n - i], matrix[n - i][n - j], matrix[n - j][i]
                matrix[i][j], matrix[j][n - i], matrix[n - i][n - j], matrix[n - j][i] = vals[3], vals[0], vals[1], vals[2]

    def rotate(self, matrix):
        '''Reverse => transpose'''
        n = len(matrix) - 1
        for i in range(len(matrix) / 2): matrix[i], matrix[n-i] = matrix[n-i], matrix[i]
        for i in range(n): for j in range(i + 1, n + 1): matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

'''735. Asteroid Collision'''
    def asteroidCollision(asteroids):
        stack = []
        while asteroids:
            stack.append(asteroids.pop())
            while len(stack) >= 2 and stack[-1] > 0 and stack[-2] < 0:
                a, b = stack.pop(), stack.pop()
                if abs(a) > abs(b):
                    stack.append(a)
                elif abs(a) < abs(b):
                    stack.append(b)
        return stack[::-1]

'''673. Number of Longest Increasing Subsequence'''
    def lengthOfLIS(nums):
        end_idx = [None] * len(nums)
        length = 0
        for i, num in enumerate(nums):
            j = 0
            while j < length and nums[end_idx[j]] < num:
                j += 1
            end_idx[j] = i
            length = max(j + 1, length)
        return length

'''331. Verify Preorder Serialization of a Binary Tree'''
    def isValidSerialization(self, preorder):
        preorder = preorder.split(',')
        diff = 1
        for node in preorder:
            diff -= 1
            if diff < 0: return False
            if node != '#': diff += 2
        return diff == 0

    def isValidSerialization(preorder):
        preorder = preorder.split(',')
        self.idx = 0
        def _isValidSerialization():
            if self.idx >= len(preorder): return False
            self.idx += 1
            if preorder[self.idx - 1] == '#': return True
            return _isValidSerialization() and _isValidSerialization()
        return _isValidSerialization() and self.idx == len(preorder)

'''209. Minimum Size Subarray Sum'''
    def minSubArrayLen(s, nums):
        min_len = total = start = 0
        for end, num in enumerate(nums):
            total += num
            while total >= s:
                min_len = min(end - start + 1, min_len or float('inf'))
                total -= nums[start]
                start += 1
        return min_len

'''688. Knight Probability in Chessboard'''
    def knightProbability(N, K, r, c):
        x_y_diff = [2, -2, 1, -1] 
        diff = [(x, y) for x in x_y_diff for y in x_y_diff if abs(x) != abs(y)]
        from collections import defaultdict
        def move(bfs):
            new_bfs = defaultdict(int)
            for (x, y, step), count in bfs.iteritems():
                for x_dif, y_dif in diff:
                    if 0 <= (x + x_dif) < N and 0 <= (y + y_dif) < N:
                        new_bfs[(x + x_dif, y + y_dif, step + 1)] += count
            return  new_bfs, sum(new_bfs.values())
        bfs = {(r, c, 0): 1}
        iteration = total = 0
        pre_count = 1
        while bfs and iteration < K:
            bfs, count = move(bfs)
            total += pre_count * len(diff) - count
            pre_count = count
            iteration += 1
        return  float(sum(bfs.values())) / len(diff) ** K

    def knightProbability(N, K, r, c):
        x_y_diff = [2, -2, 1, -1] 
        diff = [(x, y) for x in x_y_diff for y in x_y_diff if abs(x) != abs(y)]
        board = [[0] * N for _ in range(N)]
        board[r][c] = 1
        for _ in range(K):
            board = [[sum((board[x + x_dif][y + y_dif]
                           for x_dif, y_dif in diff
                           if 0 <= (x + x_dif) < N and 0 <= (y + y_dif) < N))
                      for y in range(N)]
                     for x in range(N)]
        return float(sum(map(sum, board))) / (len(diff) ** K)

'''498. Diagonal Traverse'''
    def findDiagonalOrder(matrix):
        if not matrix or not all(matrix): return []
        res = []
        def get_diag(i, j):
            new_diag = []
            while i >= 0 and j < len(matrix[0]):
                new_diag.append(matrix[i][j])
                i -= 1
                j += 1
            res.append(new_diag)
        for i in range(len(matrix)): get_diag(i, 0)
        for j in range(1, len(matrix[0])): get_diag(len(matrix) - 1, j)
        for i in range(1, len(res), 2): res[i] = list((reversed(res[i])))
        return [item for lst in res for item in lst]

'''667. Beautiful Arrangement II'''
    def constructArray(n, k):
        if k == 1: return range(1, n + 1)
        res = []
        i, j = 1, k + 1
        while i <= j:
            res.extend([j, i])
            i += 1
            j -= 1
        if len(res) != k + 1: res.pop()
        res.extend(range(k + 2, n + 1))
        return res

'''299. Bulls and Cows'''
    def getHint(secret, guess):
        a = 0
        s_counts = {}
        g_counts = {}
        for s_c, g_c in zip(secret, guess):
            if s_c == g_c: a += 1
            else:
                s_counts[s_c] = s_counts.get(s_c, 0) + 1
                g_counts[g_c] = g_counts.get(g_c, 0) + 1
        b = sum(min(s_counts.get(g_c, 0), g_counts[g_c]) for g_c in g_counts)
        return "%dA%dB" %(a, b)

'''341. Flatten Nested List Iterator'''
    class NestedIterator(object):

        def __init__(self, nestedList):
            self.stack, self.cache = [[nestedList, 0]], None

        def next(self):
            self.hasNext()
            res, self.cache = self.cache, None
            return res
        
        def hasNext(self):
            if self.cache is not None: return True
            elif not self.stack: return False
            next_lst, next_idx = self.stack[-1]
            if next_idx < len(next_lst):
                if next_lst[next_idx].isInteger():
                    self.cache = next_lst[next_idx].getInteger()
                    self.stack[-1][1] += 1
                    return True
                else:
                    self.stack[-1][1] += 1
                    self.stack.append([next_lst[next_idx].getList(), 0])
                    return self.hasNext()
            else:
                self.stack.pop()
                return self.hasNext()

'''413. Arithmetic Slices'''
    def numberOfArithmeticSlices(A):
        diffs = []
        prev = None
        for i in range(1, len(A)):
            diff = A[i] - A[i - 1]
            if not diffs or prev != diff:
                diffs.append(1)
            else: diffs[-1] += 1
            prev = diff
        return sum((n * (n - 1) / 2) for n in diffs)

'''216. Combination Sum III'''
    def combinationSum3(k, n):
        res = [[0, []]] # sum, combo
        for num in range(1, 10):
            res.extend([[num_sum + num, nums + [num]] for num_sum, nums in res if len(nums) < k])
        return [nums for num_sum, nums in res if num_sum == n and len(nums) == k]

'''33. Search in Rotated Sorted Array'''
    def search(nums, target):
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            mid = (lo + hi) / 2
            if nums[mid] == target: return mid
            elif nums[lo] <= target < nums[mid] or \
                (nums[lo] > nums[mid] and (target < nums[mid] or target >= nums[lo])): hi = mid - 1
            else: lo = mid + 1
        return -1

    def search( nums, target):
        if not nums: return -1
        # find lowest_idx
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            mid = (lo + hi) / 2
            if nums[mid] < nums[hi]: hi = mid
            else: lo = mid + 1
        lo = lo or len(nums)
        lo, hi = (0, lo) if nums[0] <= target <= nums[lo - 1] else (lo, len(nums))
        from bisect import bisect_left
        l_idx = bisect_left(nums, target, lo, hi)
        return l_idx if l_idx < len(nums) and nums[l_idx] == target else -1

'''102. Binary Tree Level Order Traversal'''
    def levelOrder(self, root):
        if not root: return []
        bfs = [root]
        res = []
        while bfs:
            res.append([node.val for node in bfs])
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return res

'''503. Next Greater Element II'''
    def nextGreaterElements(nums):
        stack, res = [], [-1] * len(nums)
        for i in range(len(nums)) * 2:
            while stack and nums[stack[-1]] < nums[i]:
                res[stack.pop()] = nums[i]
            stack.append(i)
        return res

'''646. Maximum Length of Pair Chain'''
    def findLongestChain(pairs):
        pairs.sort()
        cur = None
        count = 0
        for itv in reversed(pairs):
            if cur == None or cur[0] > itv[1]:
                cur = itv
                count += 1
        return count

'''567. Permutation in String'''
    def checkInclusion(s1, s2):
        from collections import Counter
        counts = Counter(s1)
        for i, new_char in enumerate(s2):
            counts[new_char] = counts.get(new_char, 0) - 1
            if not counts[new_char]: del counts[new_char]
            if len(s1) <= i:
                ord_char = s2[i - len(s1)]
                counts[ord_char] = counts.get(ord_char, 0) + 1
                if not counts[ord_char]: del counts[ord_char]
            if not counts: return True
        return False

'''215. Kth Largest Element in an Array'''
    def findKthLargest(nums, k):
        return sorted(nums)[len(nums) - k]

'''56. Merge Intervals'''
    def merge(intervals):
        res = []
        intervals = [[interval.start, interval.end] for interval in intervals]
        intervals.sort(reverse=True)
        while intervals:
            res.append(intervals.pop())
            if len(res) >= 2:
                [a,b], [c,d] = res[-2], res[-1]
                print [a,b], [c,d], a <= c <= b
                if a <= c <= b:
                    res.pop(); res.pop()
                    res.append([a, max(b, d)])
        return [Interval(start, end) for start, end in res]

'''63. Unique Paths II'''
    def uniquePathsWithObstacles(obstacleGrid):
        if not obstacleGrid or not obstacleGrid[0]: return 0
        dp = [[1 - item for item in row] for row in obstacleGrid]
        for i in range(1, len(dp)): dp[i][0] = min(dp[i - 1][0], dp[i][0])
        for j in range(1, len(dp[0])): dp[0][j] = min(dp[0][j - 1], dp[0][j])
        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                if dp[i][j]: dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

'''103. Binary Tree Zigzag Level Order Traversal'''
    def zigzagLevelOrder(root):
        if not root: return []
        bfs = [root]
        res = []
        while bfs:
            new_item = [node.val for node in bfs]
            res.append(new_item[::-1] if len(res) % 2 else new_item)
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return res

'''377. Combination Sum IV'''
    def combinationSum4(nums, target):
        dp = {0: 1}
        def _combinationSum4(target):
            if target in dp: return dp[target]
            res = 0
            for num in nums:
                if target - num >= 0:
                    res += _combinationSum4(target - num)
            dp[target] = res
            return res
        return _combinationSum4(target)

'''62. Unique Paths'''
    def uniquePaths(m, n):
        dp = [[1] * n for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

'''17. Letter Combinations of a Phone Number'''
    def letterCombinations(digits):
        if not digits: return []
        mappings = ['', '', 'abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
        res = ['']
        for digit in digits: res = [item + char for item in res for char in mappings[int(digit)]]
        return res

'''593. Valid Square'''
    def validSquare(p1, p2, p3, p4):
        points = [p1, p2, p3, p4]
        dists = [ ((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                 for i in range(len(points)) for j in range(i + 1, len(points))]
        from collections import Counter
        dists = Counter(dists)
        keys = dists.keys()
        return len(keys) == 2 and \
            (dists[keys[0]] == 2 or dists[keys[0]] == 4) and \
            (dists[keys[1]] == 2 or dists[keys[1]] == 4)

'''284. Peeking Iterator'''
    class PeekingIterator(object):
        def __init__(self, iterator):
            self.iterator = iterator
            self.cache = None

        def peek(self):
            if self.cache is None and self.iterator.hasNext():
                self.cache = self.iterator.next()
            return self.cache

        def next(self):
            self.peek()
            cache = self.cache
            self.cache = None
            return cache

        def hasNext(self):            self.peek()
            return not bool(self.cache is None)

'''477. Total Hamming Distance'''
    def totalHammingDistance(nums):
        nums = [[int(bool(num & (1 << i))) for i in range(31, -1, -1)] for num in nums]
        counts = [sum([num[i] for num in nums]) for i in range(31, -1, -1)]
        return sum([count * (len(nums) - count) for count in counts])

'''279. Perfect Squares'''
    def numSquares(self, n):
        bfs = [n]
        count = 0
        visited = {n}
        sq = {i: i**2 for i in range(1, int(n ** (1./2)) + 1)}
        while bfs:
            if not all(bfs): return count
            bfs = [(i - sq[j])
                      for i in bfs
                      for j in range(1, int(i ** (1./2)) + 1)
                      if (i - sq[j]) not in visited and (i - sq[j]) >= 0 and (visited.add(i - sq[j]) is None)]
            count += 1
        return 0

'''718. Maximum Length of Repeated Subarray'''
    def findLength(A, B):
        if not A or not B: return 0
        dp = [[0] * (len(B) + 1) for _ in range((len(A)) + 1)]
        for i in range(len(A)):
            for j in range(len(B)):
                if A[i] == B[j]: dp[i + 1][j + 1] = dp[i][j] + 1
        return max(item for row in dp for item in row)

'''39. Combination Sum'''
    def combinationSum(candidates, target):
        dp = [[0, []]]
        for num in candidates:
            dp = [ [cur_target + num * i, lst + ([num] * i)]
                        for cur_target, lst in dp for i in range((target - cur_target) / num + 1)]
        return [lst for num, lst in dp if num == target]

'''473. Matchsticks to Square'''
    def makesquare(nums):
        sum_of_elems = sum(nums)
        if len(nums) < 4 or sum_of_elems % 4: return False
        nums.sort(reverse=True)
        def _makesquare(pos, sums):
            if pos >= len(nums): return not any(sums)
            next_elem = nums[pos]
            visited = set()
            for i in range(len(sums)):
                if sums[i] - next_elem >= 0 and sums[i] not in visited:
                    sums[i] -= next_elem
                    if _makesquare(pos + 1, sums): return True
                    sums[i] += next_elem
                    visited.add(sums[i])
            return False
        return _makesquare(0, [sum_of_elems / 4 for _ in range(4)])

'''238. Product of Array Except Self'''
    def productExceptSelf(nums):
        dp = [1]
        for num in reversed(nums): dp.append(num * dp[-1])
        dp = dp[::-1]
        mul_so_far = 1
        res = []
        for i in range(len(nums)):
            res.append(mul_so_far * dp[i + 1])
            mul_so_far *= nums[i]
        return res

'''106. Construct Binary Tree from Inorder and Postorder Traversal'''
    def buildTree(inorder, postorder):
        def _buildTree(i_l, i_r):
            if i_l >= i_r: return
            node = TreeNode(postorder.pop())
            elem_idx = inorder.index(node.val)
            node.right = _buildTree(elem_idx + 1, i_r)
            node.left = _buildTree(i_l, elem_idx)
            return node
        return _buildTree(0, len(inorder))

'''34. Search for a Range'''
    def searchRange(nums, target):
        from bisect import bisect_left, bisect_right
        left_idx, right_idx = bisect_left(nums, target), bisect_right(nums, target) - 1
        if left_idx == -1 or left_idx >= len(nums) or nums[left_idx] != target: left_idx = -1
        if right_idx == -1 or right_idx >= len(nums) or nums[right_idx] != target: right_idx = -1
        return left_idx, right_idx

    def searchRange(nums, target):
        self.target = target
        self.low, self.high = -1, -1
        def _searchRange(low, high):
            if low >= high: return
            mid = (low + high) / 2
            if nums[mid] == self.target:
                if mid - 1 < 0 or nums[mid - 1] != self.target: self.low = mid
                else: _searchRange(low, mid)
                if mid + 1 >= len(nums) or nums[mid + 1] != self.target: self.high = mid
                else: _searchRange(mid + 1, high)
            elif nums[mid] > self.target: _searchRange(low, mid)
            elif nums[mid] < self.target: _searchRange(mid + 1, high)
        _searchRange(0, len(nums))
        return self.low, self.high

'''211. Add and Search Word - Data structure design'''
    class WordDictionary(object):

        def __init__(self):
            self.trie_tree = {}

        def addWord(self, word):
            cur = self.trie_tree
            for char in word:
                cur = cur.setdefault(char, {})
            cur[True] = word

        def search(self, word):
            stack = [self.trie_tree]
            for char in word:
                if not stack: return False
                if char == '.': stack = [cur[cur_char] for cur in stack for cur_char in cur if cur_char != True]
                else: stack = [cur[char] for cur in stack if char in cur]
            return any(cur[True] for cur in stack if True in cur)

'''208. Implement Trie (Prefix Tree)'''
    class Trie(object):

        def __init__(self): self.trie_tree = {}

        def insert(self, word):
            cur = self.trie_tree
            for char in word: cur = cur.setdefault(char, {})
            cur[True] = True
        def search(self, word):
            cur = self.trie_tree
            for char in word: 
                if char not in cur: return False
                cur = cur[char]
            return True in cur
        def startsWith(self, prefix):
            cur = self.trie_tree
            for char in prefix:
                if char not in cur: return False
                cur = cur[char]
            return True

'''380. Insert Delete GetRandom O(1)'''
    import random
    class RandomizedSet(object):
        def __init__(self):
            self.nums, self.pos = [], {}
        def insert(self, val):
            if val not in self.pos:
                self.nums.append(val)
                self.pos[val] = len(self.nums) - 1
                return True
            return False
        def remove(self, val):
            if val in self.pos:
                idx, last = self.pos[val], self.nums[-1]
                self.nums[idx], self.pos[last] = last, idx
                self.nums.pop()
                del self.pos[val]
                return True
            return False
        def getRandom(self):
            return self.nums[random.randint(0, len(self.nums) - 1)]

'''676. Implement Magic Dictionary'''
    class MagicDictionary(object):
        def __init__(self):
            self.words = {}
        def buildDict(self, dict):
            for word in dict:
                for i in range(len(word)):
                    self.words.setdefault(word[:i] + '_' + word[i + 1:], set()).add(word[i])
        def search(self, word):
            for i, char in enumerate(word):
                adjs = self.words.get(word[:i] + '_' + word[i + 1:], set())
                if adjs and ((char not in adjs) or len(adjs) >= 2): return True
            return False

'''385. Mini Parser'''
    def deserialize(s):
        i = 0
        stack = [NestedInteger()]
        while i < len(s):
            if s[i] == '[':
                ni = NestedInteger()
                stack[-1].add(ni)
                stack.append(ni)
            elif s[i].isdigit() or s[i] == '-':
                j = i
                while j < len(s) and (s[j] not in ',]'): j += 1
                ni = NestedInteger(int(s[i:j]))
                stack[-1].add(ni)
                i = j - 1
            elif s[i] == ']':
                stack.pop()
            i += 1
        return  stack[0].getList()[0]

'''278. First Bad Version'''
    def firstBadVersion(n):
        low, high = 1, n
        
        while low <= high:
            mid = (low + high) / 2
            print low, high, mid
            if isBadVersion(mid):
                if (mid - 1) < low or not isBadVersion(mid - 1): return mid
                high = mid - 1
            else:
                low = mid + 1

'''7. Reverse Integer'''
     def reverse(x):
        sign = 1 if x >= 0 else -1
        x = abs(x)
        res = 0
        while x:
            res = res * 10 + x % 10
            x /= 10
        return res * sign * (0 if res >> 31 else 1)

'''665. Non-decreasing Array'''
    def checkPossibility(nums):
        modified = False
        for i in range(1, len(nums)):
            if nums[i - 1] > nums[i]:
                if modified: return False
                if i - 2 < 0 or nums[i - 2] <= nums[i]: nums[i - 1] = nums[i]
                else: nums[i] = nums[i - 1]
                modified = True
        return True

'''189. Rotate Array'''
    def rotate(nums, k):
        def reverse(i, j):
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1
        k %= len(nums)
        reverse(len(nums) - k, len(nums) - 1)
        reverse(0, len(nums) - k - 1)
        reverse(0, len(nums) - 1)

'''475. Heaters'''
    def findRadius(houses, heaters):
        heaters.sort()
        heaters.append(float('inf'))
        cur = diff = 0
        for house in sorted(houses):
            while cur + 1 < len(heaters) and heaters[cur + 1] < house: cur += 1
            diff = max(diff, min(abs(heaters[cur] - house), abs(heaters[cur + 1] - house)))
        return diff

'''160. Intersection of Two Linked Lists'''
    def getIntersectionNode(headA, headB):
        cur_a, cur_b = headA, headB
        while cur_a and cur_b: cur_a, cur_b = cur_a.next, cur_b.next
        longer, shorter = (headA, headB) if cur_a else (headB, headA)
        cur = cur_a or cur_b
        while cur: longer, cur = longer.next, cur.next
        while longer != shorter: longer, shorter = longer.next, shorter.next
        return shorter

'''155. Min Stack'''
    class MinStack(object):

        def __init__(self):
            self.lst = []

        def push(self, x):
            cur_min = self.lst[-1][0] if self.lst else float('inf')
            self.lst.append([min(x, cur_min), x])

        def pop(self):
            self.lst.pop()

        def top(self):
            return self.lst[-1][1]

        def getMin(self):
            return self.lst[-1][0]

'''605. Can Place Flowers'''
    def canPlaceFlowers(flowerbed, n):
        count = 0
        for i, planted in enumerate(flowerbed):
            if not planted:
                if (i == 0 or not flowerbed[i-1]) and (i == (len(flowerbed) - 1) or not flowerbed[i+1]):
                    flowerbed[i] = 1
                    count += 1
            if count >= n: return True
        return False

'''67. Add Binary'''
    def addBinary(a, b):
        a, b, carry = [int(i) for i in list(a)], [int(i) for i in list(b)], 0
        res = []
        while carry or a or b:
            cur = (a.pop() if a else 0) + (b.pop() if b else 0) + carry
            carry, digit = cur / 2, cur % 2
            res.append(str(digit))
        return ''.join(list(reversed(res)))

'''234. Palindrome Linked List'''
    def isPalindrome(self, head):
        if not head or not head.next: return True
        num_elem = 0
        cur = head
        while cur:
            num_elem += 1
            cur = cur.next
        prev, cur = None, head
        mid = num_elem / 2
        for _ in range(mid):
            next_node = cur.next
            cur.next = prev
            prev, cur = cur, next_node
        if num_elem & 1: cur = cur.next
        left, right = prev, cur
        while left or right:
            if left.val != right.val: return False
            left, right = left.next, right.next
        return True

'''290. Word Pattern'''
    def wordPattern(pattern, str):
        pat_to_word = {}
        word_to_pat = {}
        words = str.split(' ')
        if len(words) != len(pattern): return False
        for i, word in enumerate(words):
            if pattern[i] in pat_to_word and pat_to_word[pattern[i]] != word or \
                word in word_to_pat and word_to_pat[word] != pattern[i]: return False
            pat_to_word.setdefault(pattern[i], word)
            word_to_pat.setdefault(word, pattern[i])
        return True

'''172. Factorial Trailing Zeroes'''
    def trailingZeroes(n):
        # factors of 5 determins number of zeros
        # ..5..10..15..20..25..30
        # ..5...5...5..5..5*5...5
        # i = 1: n/5 -> 6
        # ...1...2...3..4....5...6
        # i = 2: n/5 -> 1
        # ...1...2...3..4....1...6
        # res = 6 + 1 = 7
        res = 0
        while n:
            res += n / 5
            n /= 5
        return res

'''119. Pascal's Triangle II'''
    def getRow(rowIndex):
        if rowIndex <= 1: return [1] * (rowIndex + 1)
        cur_row = [1, 1]
        for i in range(rowIndex - 1):
            cur_row = [1] + [sum([cur_row[i-1], cur_row[i]]) for i in range(1, len(cur_row))] + [1]
        return cur_row

'''443. String Compression'''
    def compress(chars):
        i = 0
        chars.append('dummy')
        for j in range(len(chars)):
            if chars[i] == chars[j] and i != j:
                chars[j] = None
            elif  chars[i] != chars[j]:
                if j - i != 1:
                    count = str(j - i)
                    for k in range(len(count)): chars[i + k + 1] = count[k]
                i = j
        chars.pop()
        i = 0
        for j, char in enumerate(chars):
            if char is not None:
                chars[i] = char
                i += 1
        while len(chars) != i: chars.pop()
        return i
        
        434. Number of Segments in a String
    def countSegments(s):
        count = 0
        s = s.strip()
        prev = None
        for char in s:
            if prev == ' ' and char != ' ':
                count += 1
            prev = char
        if not count and s: return 1
        elif not count and not s: return 0
        elif count: return count + 1

'''541. Reverse String II'''
    def reverseStr(s, k):
        res = []
        is_even = True
        for i in range(0, len(s), k):
            part = s[i:min(i+k, len(s))]
            if is_even: part = part[::-1]
            res.extend(part)
            is_even = not is_even
        return ''.join(res)

'''653. Two Sum IV - Input is a bfs'''
    def findTarget(root, k):
        if not root: return False
        visited = set()
        bfs = [root]
        while bfs:
            for node in bfs:
                if (k - node.val) in visited: return True
                visited.add(node.val)
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return False

    def findTarget(root, k):
        def flatten(node):
            if not node:
                return []
            new_lst = flatten(node.left)
            new_lst.append(node.val)
            new_lst.extend(flatten(node.right))
            return new_lst
        lst = flatten(root)
        i, j = 0, len(lst) - 1
        while i < j:
            i_j_sum = lst[i] + lst[j]
            if i_j_sum == k:
                return True
            elif i_j_sum < k:
                i += 1
            else: j -= 1
        return False

'''70. Climbing Stairs'''
    def climbStairs(n):
        if n <= 2: return n
        prev_prev, prev = 1, 2
        cur = None
        for i in range(n - 2):
            cur = prev + prev_prev
            prev, prev_prev = cur, prev
        return cur

'''268. Missing Number'''
    def missingNumber(nums):
        n = len(nums) + 1
        return (n * (n - 1)) / 2 - sum(nums)

    def missingNumber(nums):
        for i in range(len(nums)): nums[i] += 1
        nums.append(1)
        nums.append(1)
        for i in range(len(nums) - 2): nums[abs(nums[i])] = -nums[abs(nums[i])]
        for i, num in enumerate(nums):
            if i != 0 and num > 0:
                return i - 1

'''303. Range Sum Query - Immutable'''
    class NumArray(object):
        def __init__(self, nums):
            self.l_sum = l_sum = nums
            for i in range(1, len(l_sum)): l_sum[i] = l_sum[i - 1] + l_sum[i]
        def sumRange(self, i, j):
            return self.l_sum[j] - (self.l_sum[i - 1] if i > 0 else 0)

'''633. Sum of Square Numbers'''
    def judgeSquareSum(c):
        a, b = 0, int(c ** (0.5))
        while a <= b:
            eq = a ** 2 + b ** 2
            if eq == c: return True
            elif eq < c: a += 1
            elif eq > c: b -= 1
        return False

    def judgeSquareSum(c):
        a, b = 0, int(c ** (0.5))
        dp = [0]
        inc = 1
        for i in range(b + 1):
            dp.append(inc + dp[-1])
            inc += 2
        while a <= b:
            eq = dp[a] + dp[b]
            if eq == c: return True
            elif eq < c: a += 1
            elif eq > c: b -= 1
        return False
            
            58. Length of Last Word
    def lengthOfLastWord(s):
        s = s.strip(' ')
        return len(s) - s.rfind(' ') - 1

'''226. Invert Binary Tree'''
    def invertTree(root):
        if not root: return
        bfs = [root]
        while bfs:
            for node in bfs: node.left, node.right = node.right, node.left
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return root

'''507. Perfect Number'''
    def checkPerfectNumber(num):
        if num <= 0: return False
        sqrt = num ** 0.5
        div_sum = sum(j for i in range(2, int(sqrt) + 1) if num % i == 0 for j in [i, num / i]) + 1
        if int(sqrt) == sqrt: div_sum -= sqrt
        return div_sum == num

'''26. Remove Duplicates from Sorted Array'''
    def removeDuplicates(nums):
        i = 0
        nums.append('dummy')
        for j in range(1, len(nums)):
            if nums[j - 1] != nums[j]:
                nums[i] = nums[j - 1]
                i += 1
        while i != len(nums): nums.pop()

'''572. Subtree of Another Tree'''
    def isSubtree(s, t):
        def serialize(node):
            if not node: return ''
            return '[%d,%s,%s]' %(node.val, serialize(node.left), serialize(node.right))
        return serialize(t) in serialize(s)

'''9. Palindrome Number'''
    def isPalindrome(x):
        if x < 0: return False
        y, rev = x, 0
        while y:
            rev = rev * 10 + y % 10
            y /= 10
        return rev == x

'''687. Longest Univalue Path'''
    def longestUnivaluePath(root):
        self.max_len = 0
        def _longestUnivaluePath(node):
            if not node: return 0
            left_child = right_child = 0
            left_len, right_len = _longestUnivaluePath(node.left), _longestUnivaluePath(node.right)
            if node and node.left and node.left.val == node.val: left_child = left_len + 1
            if node and node.right and node.right.val == node.val: right_child = right_len + 1
            self.max_len = max(self.max_len, left_child + right_child)
            return max(left_child, right_child)
        _longestUnivaluePath(root)
        return self.max_len

'''669. Trim a Binary Search Tree'''
    def trimBST(root, L, R):
        def _trimBST(node, L, R):
            if not node: return
            elif node.val < L: return _trimBST(node.right, L, R)
            elif node.val > R: return _trimBST(node.left, L, R)
            else:
                node.left = _trimBST(node.left, L, R)
                node.right = _trimBST(node.right, L, R)
                return node
        return _trimBST(root, L, R)

'''463. Island Perimeter'''
    def islandPerimeter(grid):
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]:
                    adjs = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                    for adj in adjs:
                        if not (0 <= adj[0] < len(grid) and 0 <= adj[1] < len(grid[0])) or \
                            not grid[adj[0]][adj[1]]:
                            count += 1
        return count

'''594. Longest Harmonious Subsequence'''
    def findLHS(nums):
        from collections import Counter
        counts = Counter(nums)
        return max([counts[x] + counts[x + 1] for x in counts if x + 1 in counts] or [0])

'''674. Longest Continuous Increasing Subsequence'''
    def findLengthOfLCIS(nums):
        nums.append(float('-inf'))
        i = max_len = 0
        for j in range(1, len(nums)):
            if nums[j-1] >= nums[j]:
                max_len = max(max_len, j - i)
                i = j
        return max_len

'''551. Student Attendance Record I'''
    def checkRecord(s): return (s.count('A') <= 1) and ('LLL' not in s)

'''733. Flood Fill'''
    def floodFill(image, sr, sc, newColor):
        stack = [(sr, sc)]
        old_col, new_col = image[sr][sc], newColor
        if old_col == new_col: return image
        while stack:
            i, j = stack.pop()
            image[i][j] = new_col
            stack.extend([(x, y) for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] \
                        if 0 <= x < len(image) and 0 <= y < len(image[0]) and image[x][y] == old_col])
        return image

'''88. Merge Sorted Array'''
    def merge(self, nums1, m, nums2, n):\
        i, j = m - 1, n - 1
        for k in range(i + j + 1, -1, -1):
            if j < 0: nums1[k], i = nums1[i], i - 1
            elif i < 0: nums1[k], j = nums2[j], j - 1
            elif nums1[i] > nums2[j]: nums1[k], i = nums1[i], i - 1
            else: nums1[k], j = nums2[j], j - 1

'''326. Power of Three'''
    def isPowerOfThree(n):
        # import sys
        # power = 1
        # while (3 ** (power + 1)) <= sys.maxint:
        #     power += 1
        # print 3 ** power
        return (n > 0) and (4052555153018976267 % n == 0)

'''108. Convert Sorted Array to Binary Search Tree'''
    def sortedArrayToBST(nums):
        def _sortedArrayToBST(i, j):
            if i >= j: return
            mid = (i + j) / 2
            node = TreeNode(nums[mid])
            node.left = _sortedArrayToBST(i, mid)
            node.right = _sortedArrayToBST(mid + 1, j)
            return node

        return _sortedArrayToBST(0, len(nums))

'''606. Construct String from Binary Tree'''
    def tree2str(t):
        def _tree2str(node):
            if not node: return ''
            left = _tree2str(node.left)
            right = _tree2str(node.right)
            if not left and not right: return str(node.val)
            elif not right: return '%d(%s)' %(node.val, left)
            else: return '%d(%s)(%s)' %(node.val, left, right)
        return _tree2str(t)

'''690. Employee Importance'''
    def getImportance(employees, id):
        id_to_person = {person.id: person for person in employees}
        adjs = {person: [id_to_person[adj_id] for adj_id in person.subordinates] for person in employees}
        visited = set()
        bfs = [id_to_person[id]]
        importance = 0
        while bfs:
            importance += sum(person.importance for person in bfs)
            bfs = [adj for person in bfs for adj in adjs[person] if adj not in visited and (not visited.add(adj))]
        return importance

'''125. Valid Palindrome'''
    def isPalindrome(s):
        s = [char.lower() for char in s if char.isalpha() or char.isdigit()]
        return s == s[::-1]

'''695. Max Area of Island'''
    def maxAreaOfIsland(grid):
        def flip(i, j):
            if not grid[i][j]: return 0
            count = 0
            to_visit = {(i, j)}
            while to_visit:
                count += len(to_visit)
                for i, j in to_visit: grid[i][j] = None
                to_visit = set((x, y) for i, j in to_visit
                                      for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                                      if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y])
            return count
        
        max_area = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                max_area = max(max_area, flip(i, j))
        return max_area

'''141. Linked List Cycle'''
    def hasCycle(head):
        visited = set()
        cur = head
        while cur:
            if cur in visited: return True
            visited.add(cur)
            cur = cur.next
        return False

'''204. Count Primes'''
    def countPrimes(n):
        if n <= 2: return 0
        primes = [True] * (n)
        primes[0] = primes[1] = False
        for i in range(2, n):
            if primes[i]:
                for j in range(i, n, i):
                    if j != i: primes[j] = False
        return primes.count(True)

'''14. Longest Common Prefix'''
    def longestCommonPrefix(strs):
        if not strs: return ''
        longest_prefix = None
        for str_item in strs:
            if longest_prefix is None:
                longest_prefix = str_item
            else:
                min_len = min(len(longest_prefix), len(str_item))
                longest_prefix = longest_prefix[:min_len]
                for i in range(min_len):
                    if longest_prefix[i] != str_item[i]:
                        longest_prefix = longest_prefix[:i]
                        break
        return longest_prefix

'''367. Valid Perfect Square'''
    def isPerfectSquare(num):
        diff = 3
        sq = 1
        while sq < num:
            sq += diff
            diff += 2
        return sq == num

'''504. Base 7'''
    def convertToBase7(num):
        base = 7
        res = ""
        neg_prefix = '-' if num < 0 else ''
        num = abs(num)
        while num != 0:
            num, mod = divmod(num, base)
            res = str(mod) + res
        return (neg_prefix + res) or '0'

'''345. Reverse Vowels of a String'''
    def reverseVowels(s):
        vowels = [char for char in s if char in 'aeiouAEIOU']
        return ''.join([ (char if char not in 'aeiouAEIOU' else vowels.pop()) for char in s])

'''83. Remove Duplicates from Sorted List'''
    def deleteDuplicates(head):
        cur = head
        res = res_head = ListNode('dummy')
        while cur:
            if cur.val != res.val:
                res.next = cur
                res = res.next
            cur = cur.next
        res.next = None
        return res_head.next

'''643. Maximum Average Subarray I'''
    def findMaxAverage(nums, k):
        for i in range(1, len(nums)): nums[i] += nums[i - 1]
        max_avg = float('-inf')
        for i in range(k - 1, len(nums)):
            left, right = (0 if i - k < 0 else nums[i - k]), float(nums[i])
            max_avg = max(max_avg, (right - left) / k)
        return max_avg

'''191. Number of 1 Bits'''
    def hammingWeight(n):
        count = 0
        while n:
            count += 1 & n
            n = n >> 1
        return count

'''645. Set Mismatch'''
    def findErrorNums(nums):
        for num in nums:
            if nums[abs(num) - 1] < 0: 
                dup = abs(num)
                break
            nums[abs(num) - 1] = -nums[abs(num) - 1]
        for i in range(len(nums)): nums[i] = abs(nums[i])
        n = len(nums)
        return dup, ((n * (n + 1) / 2)) - sum(nums) + dup

'''53. Maximum Subarray'''
    def maxSubArray(nums):
        max_sum = float('-inf')
        max_ending_here = float('-inf')
        for i in range(len(nums)):
            max_ending_here = max(max_ending_here, 0) + nums[i]
            max_sum = max(max_ending_here, max_sum)
        return max_sum

'''459. Repeated Substring Pattern'''
    def repeatedSubstringPattern(s): return s in (s + s)[1:-1]

'''27. Remove Element'''
    def removeElement(nums, val):
            i = 0
            for j, num in enumerate(nums):
                if val != num:
                    nums[i] = nums[j]
                    i += 1
            while i != len(nums): nums.pop()
            return i
                     

'''107. Binary Tree Level Order Traversal II'''
    def levelOrderBottom(root):
        if not root: return []
        lvl_tra = []
        bfs = [root]
        while bfs:
            lvl_tra.append([node.val for node in bfs])
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return lvl_tra[::-1]

'''724. Find Pivot Index'''
    def pivotIndex(nums):
        right_sums = nums + [0]
        for i in range(len(right_sums) - 2, -1, -1): right_sums[i] += right_sums[i + 1]
        left_sum = 0
        for i in range(len(nums)):
            if left_sum == right_sums[i + 1]: return i
            left_sum += nums[i]
        return -1
        
        110. Balanced Binary Tree
    def isBalanced(root):
        def _isBalanced(node):
            if not node: return 0
            left = _isBalanced(node.left)
            if left == -1: return -1
            right = _isBalanced(node.right)
            return -1 if (abs(left - right) > 1 or right == -1) else max(left, right) + 1
        return _isBalanced(root) != -1

'''671. Second Minimum Node In a Binary Tree'''
    def findSecondMinimumValue(root):
        def next_mins(node, root_val, res=[]):
            if not node: return res
            if node.val != root_val: res.append(node.val)
            else:
                next_mins(node.left, root_val, res)
                next_mins(node.right, root_val, res)
                return res
        if not root: return -1
        next_mins = next_mins(root, root.val)
        return min(next_mins) if next_mins else -1

'''415. Add Strings'''
    def addStrings(num1, num2):
        def convert_to_int(num):
            res = 0
            for digit in num:
                res *= 10
                res += ord(digit) - ord('0')
            return res
        return str(convert_to_int(num1) + convert_to_int(num2))

'''121. Best Time to Buy and Sell Stock'''
    def maxProfit(prices):
        max_profit = max_so_far = 0
        for price in reversed(prices):
            max_profit = max(max_so_far - price, max_profit)
            max_so_far = max(max_so_far, price)
        return max_profit

'''389. Find the Difference'''
    def findTheDifference(s, t):
        return chr(reduce(operator.xor, [ord(char) for char in s + t], 0))

'''717. 1-bit and 2-bit Characters'''
    def isOneBitCharacter(bits):
        i = 0
        is_one_bit = False
        while i < len(bits):
            is_one_bit = not bits[i]
            if not bits[i]: i += 1
            else: i += 2
        return is_one_bit

    def isOneBitCharacter(bits):
        import re
        return re.findall('(1.|0)', reduce(lambda x, y : x + str(y), bits, ''))[-1] == '0'

'''492. Construct the Rectangle'''
    def constructRectangle(area):
        i = j = int(area ** 0.5)
        while (i * j) != area:
            if i * j > area: i -= 1
            else: j += 1
        return j, i

'''520. Detect Capital'''
    def detectCapitalUse(word):
        is_chars_upper = [(char == char.upper()) for char in reversed(word)]
        first_char_upper = is_chars_upper.pop()
        return first_char_upper and all(is_chars_upper) or not any(is_chars_upper)

'''171. Excel Sheet Column Number'''
    def convertToTitle(n):
        res = []
        mapping = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        while n != 0:
            res.append(mapping[(n - 1) % 26])
            n = (n - 1) / 26
            
        return ''.join(res[::-1])

'''598. Range Addition II'''
    def maxCount(m, n, ops):
        max_x = min([op[0] for op in ops if op[0]] or [0])
        max_y = min([op[1] for op in ops if op[1]] or [0])
        return (max_x * max_y) or (m * n)

'''202. Happy Number'''
    def isHappy(n):
        visited = set()
        while n != 1:
            if n in visited: return False
            visited.add(n)
            x, n = n, 0
            while x:
                n += (x % 10) ** 2
                x /= 10
        return True

'''28. Implement strStr()'''
    def strStr(haystack, needle):
        for i in range(0, len(haystack) - len(needle) + 1):
            if haystack[i: i + len(needle)] == needle:
                return i
        return -1

'''680. Valid Palindrome II'''
    def validPalindrome(self, s):
        for i in range(len(s) / 2):
            j = len(s) - i - 1
            if s[i] != s[j]:
                s1 = s[:i] + s[i + 1:]
                s2 = s[:j] + s[j + 1:]
                return s1 == s1[::-1] or s2 == s2[::-1]
        return True

'''1. Two Sum'''
    def twoSum(nums, target):
        seen = {}
        for i, num in enumerate(nums):
            diff = target - num
            if diff in seen: return [seen[diff], i]
            seen[num] = i

'''219. Contains Duplicate II'''
    def containsNearbyDuplicate(nums, k):
        past = {}
        for i, num in enumerate(nums):
            if num in past and i - past[num] <= k: return True
            past[num] = i
        return False

'''203. Remove Linked List Elements'''
    def removeElements(head, val):
        res = res_cur = ListNode('dummy')
        cur = head
        while cur:
            if cur.val == val:
                cur = cur.next
            else:
                res_cur.next = cur
                cur = cur.next
                res_cur = res_cur.next
                res_cur.next = None
        return res.next

'''617. Merge Two Binary Trees'''
    def mergeTrees(t1, t2):
        def _merge_nodes(node1, node2):
            if not (node1 and node2): return node1 or node2
            node1.val += node2.val
            node1.left = _merge_nodes(node1.left, node2.left)
            node1.right = _merge_nodes(node1.right, node2.right)
            return node1
        return _merge_nodes(t1, t2)
            
            501. Find Mode in Binary Search Tree
    def findMode(root):
        counts = {}
        if not root: return []
        bfs = [root]
        while bfs:
            for node in bfs:
                counts[node.val] = counts.get(node.val, 0) + 1
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        max_freq = max(counts.values())
        return [key for key, freq in counts.iteritems() if freq == max_freq]

'''235. Lowest Common Ancestor of a Binary Search Tree'''
    def lowestCommonAncestor(root, p, q):
        def _lowestCommonAncestor(node):
            if not node or node == p or node == q: return node
            left, right = _lowestCommonAncestor(node.left), _lowestCommonAncestor(node.right)
            return node if (left and right) else (left or right)
        return _lowestCommonAncestor(root)

'''657. Judge Route Circle'''
    def judgeCircle(moves):
        from collections import Counter
        counts = Counter(moves)
        return counts.get('U', 0) == counts.get('D', 0) and counts.get('L', 0) == counts.get('R', 0)

'''198. House Robber'''
    def rob(nums):
        for i in range(1, len(nums)):
            nums[i] = max([nums[i - 1], nums[i]] if i - 2 < 0 else [nums[i - 1], nums[i - 2] + nums[i]])
        return nums[-1] if nums else 0

'''342. Power of Four'''
    def isPowerOfFour(num):
        # mask = 0
        # while (mask << 2 | 1) < ((1 << 31) - 1):
        #     mask = mask << 2 | 1
        # print mask
        return (num & (num - 1)) == 0 and bool(1431655765 & num)

'''101. Symmetric Tree'''
    def isSymmetric(root):
        def _isSymmetric(t1, t2):
            if not t1 and not t2: return True
            elif (not t1 and t2) or (t1 and not t2): return False
            return t1.val == t2.val and \
                    _isSymmetric(t1.left, t2.right) and \
                    _isSymmetric(t1.right, t2.left)
        if not root: return True
        return _isSymmetric(root.left, root.right)

'''682. Baseball Game'''
    def calPoints(ops):
        stack = []
        for item in ops:
            if item == 'D': stack.append(stack[-1] * 2)
            elif item == '+': stack.append(stack[-1] + stack[-2])
            elif item == 'C': stack.pop()
            else: stack.append(int(item))
        return sum(stack)

'''697. Degree of an Array'''
    def findShortestSubArray(nums):
        from collections import Counter
        counts = Counter(nums)
        max_freq = max(counts.values())
        num_to_start_end = {key:(len(nums), 0) for key, val in counts.iteritems() if val == max_freq}
        for i, num in enumerate(nums):
            if num in num_to_start_end:
                start, end = num_to_start_end[num]
                num_to_start_end[num] = (min(i, start), max(i, end))
        return min([end - start + 1 for start, end in num_to_start_end.values()])

'''461. Hamming Distance'''
    def hammingDistance(x, y):
        count = 0
        while x or y:
            if (x & 1) != (y & 1): count += 1
            x >>= 1
            y >>= 1
        return count

'''401. Binary Watch'''
    def readBinaryWatch(num):
        def count_ones(num):
            count = 0
            while num != 0:
                count += num % 2
                num /= 2
            return count
        return [
            "%d:%02d" %(hr, m)
            for hr in range(12)
            for m in range(60)
            if count_ones(hr) + count_ones(m) == num
        ]

'''387. First Unique Character in a String'''
    def firstUniqChar(s):
        from collections import Counter
        counts = Counter(s)
        for i, char in enumerate(s):
            if counts[char] == 1:
                return i
        return -1

'''66. Plus One'''
    def plusOne(digits):
        for i in range(len(digits) - 1, -1, -1):
            if digits[i] < 9:
                digits[i] += 1
                return digits
            else: digits[i] = 0
        digits.insert(0, 1)
        return digits

'''231. Power of Two'''
    def isPowerOfTwo(n):
        return (n > 0) and (n & (n - 1)) == 0

'''575. Distribute Candies'''
    def distributeCandies(candies):
        return min(len(candies)/2, len(set(candies)))

'''455. Assign Cookies'''
    def findContentChildren(g, s):
        g.sort(reverse=True)
        s.sort(reverse=True)
        count = 0
        while s and g:
            cookie = s.pop()
            if cookie >= g[-1]:
                g.pop()
                count += 1
        return count

'''496. Next Greater Element I'''
    def nextGreaterElement(findNums, nums):
        return [next((num for num in nums[nums.index(num_f):] if num > num_f), -1) for num_f in findNums]

'''485. Max Consecutive Ones'''
    def findMaxConsecutiveOnes(nums):
        max_len = 0
        for j, val in enumerate(nums):
            if val:
                if j - 1 < 0 or not (nums[j - 1]): i = j
                max_len = max(max_len, j - i + 1)
        return max_len

'''637. Average of Levels in Binary Tree'''
    def averageOfLevels(root):
        if not root: return []
        avgs = []
        bfs = [root]
        while bfs:
            vals = [node.val for node in bfs]
            avgs.append(float(sum(vals)) / len(vals))
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return avgs

'''242. Valid Anagram'''
    def isAnagram(self, s, t):
        from collections import Counter
        return Counter(s) == Counter(t)

    def isAnagram(s, t):
        alphabet_count = 26
        counts = [0] * alphabet_count
        for char in s: counts[ord(char) % alphabet_count] += 1
        for char in t: counts[ord(char) % alphabet_count] -= 1
        return not any(counts)
            

'''566. Reshape the Matrix'''
    def matrixReshape(nums, r, c):
        if r * c != len(nums) * len(nums[0]): return nums
        res = [num for row in nums for num in row]
        return [res[i:i+c] for i in range(0, len(res), c)]

'''500. Keyboard Row'''
    def findWords(words):
        keyboard = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
        res = []
        for word in words:
            char = word[0].lower()
            for row in keyboard:
                if char in row:
                    if all(char.lower() in row for char in word):
                        res.append(word)
                        break
        return res

'''104. Maximum Depth of Binary Tree'''
    def maxDepth(root):
        if not root: return 0
        depth = 0
        bfs = [root]
        while bfs:
            depth += 1
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return depth

'''538. Convert bfs to Greater Tree'''
    def convertBST(root):
        self.cur_sum = 0
        def _convertBST(node):
            if not node: return
            _convertBST(node.right)
            node.val = self.cur_sum = node.val + self.cur_sum
            _convertBST(node.left)
        _convertBST(root)
        return root

'''521. Longest Uncommon Subsequence I'''
    def findLUSlength(a, b):
        return -1 if a == b else max(len(a), len(b))

'''530. Minimum Absolute Difference in bfs'''
    def getMinimumDifference(root):
        if not root: return
        bfs = [root]
        vals = []
        while bfs:
            vals.extend([node.val for node in bfs])
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        vals.sort()
        return min(vals[i] - vals[i - 1] for i in range(1, len(vals)))

'''728. Self Dividing Numbers'''
    def selfDividingNumbers(left, right):
        res = []
        for i in range(left, right + 1):
            x = i
            is_div = True
            while is_div and x:
                x, mod = divmod(x, 10)
                if not mod or i % mod != 0: is_div = False
            if is_div: res.append(i)
        return res

'''38. Count and Say'''
    def countAndSay(n):
        cur = ['1']
        for i in range(n - 1):
            stack = []
            for elem in cur:
                if not stack or stack[-1][1] != elem: stack.append([1, elem])
                else: stack[-1][0] += 1
            cur = [str(i) for pair in stack for i in pair]
        return ''.join(cur)

'''217. Contains Duplicate'''
    def containsDuplicate(nums):
        return len(nums) != len(set(nums))

'''661. Image Smoother'''
    def imageSmoother(M):
        if not M or not any(M): return M
        res = [[0] * len(M[0]) for _ in range(len(M))]
        for i in range(len(res)):
            for j in range(len(res[0])):
                dif = [-1, 0, 1]
                adjs = [M[x][y]
                        for x, y in
                        [(i + dif_x, j + dif_y) for dif_x in dif for dif_y in dif]
                        if 0 <= x < len(res) and 0 <= y < len(res[0])]
                res[i][j] = sum(adjs) / len(adjs)
        return res

'''506. Relative Ranks'''
    def findRelativeRanks(nums):
        sorted_scores = sorted(nums, reverse=True)
        str_scores = ["Gold Medal", "Silver Medal", "Bronze Medal"]
        str_scores = (str_scores + [str(i) for i in range(4, len(nums) + 1)])[:len(nums)]
        score_to_str = dict(zip(sorted_scores, str_scores))
        return [score_to_str[score] for score in nums]

'''206. Reverse Linked List'''
    def reverseList(head):
        cur = head
        prev = None
        while cur:
            next_node = cur.next
            cur.next = prev
            prev, cur = cur, next_node
        return prev

'''720. Longest Word in Dictionary'''
    def longestWord(words):
        words_by_len = [set([''])]
        for word in words:
            while len(word) >= len(words_by_len): words_by_len.append(set())
            words_by_len[len(word)].add(word)

        for i in range(1, len(words_by_len)):
            prev = words_by_len[i - 1]
            cur = words_by_len[i]
            for word in set(words_by_len[i]):
                if word[:-1] not in prev:
                    cur.remove(word)
            if not cur: return min(prev)
        return min(cur)

'''344. Reverse String'''
    def reverseString(s): return s[::-1]

'''374. Guess Number Higher or Lower'''
    def guessNumber(n):
        low, high = 1, n
        while low <= high:
            mid = (low + high) / 2
            print low, high, mid
            if guess(mid) == 0: return mid
            elif guess(mid) > 0: low = mid + 1
            else: high = mid - 1

'''437. Path Sum III'''
    def pathSum(root, sum):
        def _pathSum(node, target, sums_count={0: 1}, so_far=0):
            if not node: return 0
            so_far += node.val
            count = sums_count.get(so_far - target, 0)
            sums_count.setdefault(so_far, 0)
            sums_count[so_far] += 1
            count += _pathSum(node.left, target, sums_count, so_far)
            count += _pathSum(node.right, target, sums_count, so_far)
            sums_count[so_far] -= 1
            if so_far in sums_count and not sums_count[so_far]: del sums_count[so_far]
            return count
        return _pathSum(root, sum)

'''237. Delete Node in a Linked List'''
    def deleteNode(node):
        node.val = node.next.val
        node.next = node.next.next

'''412. Fizz Buzz'''
    def fizzBuzz(n):
        return [('Fizz' * (not i % 3) + 'Buzz' * (not i % 5)) or str(i) for i in range(1, n + 1)]

'''167. Two Sum II - Input array is sorted'''
    def twoSum(numbers, target):
        i, j = 0, len(numbers) - 1
        while i < j:
            pair_sum = numbers[i] + numbers[j]
            if pair_sum == target: return i + 1, j + 1
            elif pair_sum < target: i += 1
            else: j -= 1

'''628. Maximum Product of Three Numbers'''
    def maximumProduct(nums):
        max1 = max2 = max3 = float('-inf')
        min1 = min2 = float('inf')
        for num in nums:
            
            if num > max1: max1, max2, max3 = num, max1, max2
            elif num > max2: max2, max3 = num, max2
            elif num > max3: max3 = num
                
            if num < min1: min1, min2 = num, min1
            elif num < min2: min2 = num
        return max(max1 * max2 * max3, min1 * min2 * max1)

    def maximumProduct(nums):
        max3, min2 = [], []
        for num in nums:
            if (not max3.append(num)) and len(max3) == 4: max3.remove(min(max3))
            if (not min2.append(num)) and len(min2) == 3: min2.remove(max(min2))
        def prod(lst): return reduce(lambda x, y: x * y, lst, 1)
        return max(prod(max3), prod(min2 + [max(max3)]))

'''599. Minimum Index Sum of Two Lists'''
    def findRestaurant(list1, list2):
        rest_to_idx_2 = {rest: i for i, rest in enumerate(list2)}
        min_dist, min_rests = float('inf'), []
        for i, rest in enumerate(list1):
            if rest in rest_to_idx_2:
                fav_sum = rest_to_idx_2[rest] + i
                if fav_sum < min_dist: min_dist, min_rests = fav_sum, [rest]
                elif fav_sum == min_dist: min_rests.append(rest)
        return min_rests

'''111. Minimum Depth of Binary Tree'''
    def minDepth(root):
        if not root: return 0
        depth, bfs = 0, [root]
        while bfs:
            depth += 1
            if any(node for node in bfs if not node.left and not node.right): return depth
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]

'''453. Minimum Moves to Equal Array Elements'''
    def minMoves(nums): return sum(nums) - len(nums) * min(nums)

'''383. Ransom Note'''
    def canConstruct(ransomNote, magazine):
        counts = [0] * 26
        for char in magazine: counts[ord(char) % len(counts)] += 1
        for char in ransomNote: counts[ord(char) % len(counts)] -= 1
        return all(count >= 0 for count in counts)

'''35. Search Insert Position'''
    def searchInsert(nums, target):
        from bisect import bisect_left
        return bisect_left(nums, target)

'''100. Same Tree'''
    def isSameTree(p, q):
        def _isSameTree(node1, node2):
            if not any([node1, node2]): return True
            elif not all([node1, node2]): return False
            return node1.val == node2.val and \
                    _isSameTree(node1.left, node2.left) and \
                    _isSameTree(node1.right, node2.right)
        return _isSameTree(p, q)

'''122. Best Time to Buy and Sell Stock II'''
    def maxProfit(prices):
        profit = 0
        for i in range(1, len(prices)):
            profit += prices[i] - prices[i - 1] if prices[i] > prices[i - 1] else 0
        return profit

'''136. Single Number'''
    def singleNumber(nums): return reduce(lambda x, y: x ^ y, nums, 0)

'''693. Binary Number with Alternating Bits'''
    def hasAlternatingBits(self, n):
        toggle = n & 1
        while n:
            n, is_one = divmod(n, 2)
            if toggle and not is_one or not toggle and is_one: return False
            toggle = not toggle
        return True

'''190. Reverse Bits'''
    def reverseBits(n):
        res = 0
        for i in range(32):
            n, mod = divmod(n, 2)
            res = (res << 1) | mod
        return res

'''21. Merge Two Sorted Lists'''
    def mergeTwoLists(l1, l2):
        dummy = l = ListNode('dummy')
        while l1 or l2:
            if l1 and l2: l1, l2 = (l1, l2) if l1.val < l2.val else (l2, l1)
            else: l1, l2 = l1 or l2, None
            l.next = l1
            l, l1 = l.next, l1.next
        return dummy.next

'''447. Number of Boomerangs'''
    def numberOfBoomerangs(points):
        res = 0
        for i, [x, y] in enumerate(points):
            dist_to_point = {}
            for j, [adj_x, adj_y] in enumerate(points):
                if i != j:
                    key = (x - adj_x) ** 2 + (y - adj_y) ** 2
                    dist_to_point[key] = dist_to_point.setdefault(key, 0) + 1
            res += sum([val * (val - 1) for val in dist_to_point.values()])
        return res

'''532. K-diff Pairs in an Array'''
    def findPairs(nums, k):
        from collections import Counter
        counts = Counter(nums)
        res = 0
        for val in counts:
            if k > 0 and val + k in counts or \
                not k and counts[val] > 1:
                res += 1
        return res

'''543. Diameter of Binary Tree'''
    def diameterOfBinaryTree(root):
        self.max = 0
        def _longest_len(node):
            if not node: return 0
            l, r = _longest_len(node.left), _longest_len(node.right)
            self.max = max(self.max, l + r)
            return max(l, r) + 1
        _longest_len(root)
        return self.max

'''563. Binary Tree Tilt'''
    def findTilt(root):
        self.tilt = 0
        def get_sum_update_tilt(node):
            if not node: return 0
            l, r = get_sum_update_tilt(node.left), get_sum_update_tilt(node.right)
            self.tilt += abs(l - r)
            return l + r + node.val
        get_sum_update_tilt(root)
        return self.tilt

'''561. Array Partition I'''
    def arrayPairSum(nums):
        nums.sort()
        return sum(nums[::2])

'''205. Isomorphic Strings'''
    def isIsomorphic(s, t):
        s_to_t = {}
        for i in range(len(s)):
            s_to_t[ord(s[i])] = ord(t[i])
            s_to_t[ord(t[i]) << 10] = ord(s[i])
        for i in range(len(s)):
            if not(s_to_t[ord(s[i])] == ord(t[i]) and s_to_t[ord(t[i]) << 10] == ord(s[i])): return False
        return True

'''258. Add Digits'''
    def addDigits(num):
        while (num / 10):
            n_num = 0
            while num:
                num, mod = divmod(num, 10)
                n_num += mod
            num = n_num
        return num

'''476. Number Complement'''
    def findComplement(self, num):
        mask = 0
        while mask & num != num: mask = (mask << 1) | 1
        return num ^ mask

'''581. Shortest Unsorted Continuous Subarray'''
    def findUnsortedSubarray(nums):
        sorted_nums = sorted(nums)
        i, j = 0, len(nums) - 1
        while i < j:
            if nums[i] == sorted_nums[i]: i += 1
            elif nums[j] == sorted_nums[j]: j -= 1
            else: return j - i + 1
        return 0

'''404. Sum of Left Leaves'''
    def sumOfLeftLeaves(root):
        if not root: return 0
        left_leave_sum = 0
        bfs = [root]
        while bfs:
            left_leave_sum += sum(node.left.val for node in bfs if node.left and not node.left.right and not node.left.left)
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return left_leave_sum

'''257. Binary Tree Paths'''
    def binaryTreePaths(root):
        if not root: return []
        bfs = [(root, str(root.val))]
        paths = []
        while bfs:
            paths.extend([path for node, path in bfs if not node.left and not node.right])
            bfs = [(kid, "%s->%s" %(path, str(kid.val))) for node, path in bfs for kid in [node.left, node.right] if kid]
        return paths

'''349. Intersection of Two Arrays'''
    def intersection(self, nums1, nums2): return list(set(nums1) & set(nums2))

'''169. Majority Element'''
    def majorityElement(nums):
        count, most_freq_elem = 0, None
        for num in nums:
            if most_freq_elem is None:
                count, most_freq_elem = 1, num
            elif num != most_freq_elem:
                count -= 1
                if count == 0:
                    count, most_freq_elem = 1, num
            elif num == most_freq_elem:
                count += 1
        return most_freq_elem

'''686. Repeated String Match'''
    def repeatedStringMatch(A, B):
        for i in range(3):
            times = len(B) / len(A) + i
            if B in A * times:
                return times
        return -1
        112. Path Sum
    def hasPathSum(root, total):
        cur_path = []
        def _pathSum(node, sum_from_root):
            if not node:
                return False
            sum_from_root += node.val
            cur_path.append(node.val)
            if sum_from_root == total and not node.left and not node.right:
                return True
            res = _pathSum(node.left, sum_from_root) or _pathSum(node.right, sum_from_root)
            cur_path.pop()
            return res
        return _pathSum(root, 0)

'''168. Excel Sheet Column Title'''
    def convertToTitle(n):
        res = []
        mapping = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        while n != 0:
            res.append(mapping[(n - 1) % 26])
            n = (n - 1) / 26
        return ''.join(list(reversed(res)))

'''283. Move Zeroes'''
    def moveZeroes(nums):
        i = 0
        for j, num in enumerate(nums):
            if num != 0:
                nums[i] = num
                i += 1
        for j in range(i, len(nums)):
            nums[j] = 0

'''414. Third Maximum Number'''
    def thirdMax(nums):
        if not nums: return
        max_three = []
        for num in set(nums):
            max_three.append(num)
            if len(max_three) > 3:
                max_three.remove(min(max_three))
        return min(max_three) if len(max_three) >= 3 else max(max_three)

'''20. Valid Parentheses'''
    def isValid(s):
        stack = []
        closing_to_opening = {')':'(', '}':'{', ']':'['}
        for char in s:
            if char in '()[]{}':
                if char in closing_to_opening:
                    if not stack or stack[-1] != closing_to_opening[char]:
                        return False
                    stack.pop()
                else:
                    stack.append(char)
        return not stack

'''557. Reverse Words in a String III'''
    def reverseWords(s): return ' '.join([word[::-1] for word in s.split(' ')])

'''438. Find All Anagrams in a String'''
    def findAnagrams(s, p):
        from collections import Counter
        p_occur = Counter(p)
        s_part_occur = Counter(s[:len(p)])
        res = []
        for i in range(len(s) - len(p)):
            if p_occur == s_part_occur:
                res.append(i)
            s_part_occur[s[i]] -= 1
            if not s_part_occur[s[i]]: del s_part_occur[s[i]]
            s_part_occur.setdefault(s[i + len(p)], 0)
            s_part_occur[s[i + len(p)]] += 1
        if p_occur == s_part_occur: res.append(len(s) - len(p))
        return res

'''448. Find All Numbers Disappeared in an Array'''
    def findDisappearedNumbers(nums):
        for num in nums: nums[abs(num) - 1] = -1 * abs(nums[abs(num) - 1])
        return [i + 1 for i, num in enumerate(nums) if num > 0]

'''118. Pascal's Triangle'''
    def generate(numRows):
        tri = []
        for i in range(1, numRows + 1):
            for j in range(i):
                if j == 0: tri.append([1])
                elif j == i - 1: tri[-1].append(1)
                else: tri[-1].append(sum(tri[-2][j - 1: j + 1]))
        return tri

'''350. Intersection of Two Arrays II'''
    def intersect(nums1, nums2):
        from collections import Counter
        counts1 = Counter(nums1)
        res = []
        for num in nums2:
            if counts1.get(num, 0):
                res.append(num)
                counts1[num] -= 1
        return res

'''696. Count Binary Substrings'''
    def countBinarySubstrings(s):
        s = map(len, s.replace('01', '0 1').replace('10', '1 0').split())
        return sum([min(s[i], s[i + 1]) for i in range(len(s) - 1)])

'''292. Nim Game'''
    def canWinNim(n): return  (n % 4) != 0

'''409. Longest Palindrome'''
    def longestPalindrome(s):
        from collections import Counter
        odds_len = sum(count & 1 for count in Counter(s).values())
        return len(s) - odds_len + bool(odds_len)

'''232. Implement Queue using Stacks'''
    class MyQueue(object):

        def __init__(self):
            self.incoming = []
            self.outgoing = []

        def push(self, x): self.incoming.append(x)
        
        def _move_to_outgoing(self):
            if not self.outgoing:
                while self.incoming:
                    self.outgoing.append(self.incoming.pop())
            return self.outgoing

        def pop(self): return self._move_to_outgoing().pop()

        def peek(self): return self._move_to_outgoing()[-1]

        def empty(self): return not (self.incoming or self.outgoing)

'''225. Implement Stack using Queues'''
    class MyStack(object):
        def __init__(self):
            import Queue
            self.pri, self.sec = Queue.PriorityQueue(), Queue.PriorityQueue()

        def push(self, x):
            self.pri.put(x)

        def pop(self):
            while True:
                tmp = self.pri.get() if not self.pri.empty() else None
                if not self.pri.empty(): self.sec.put(tmp)
                else:
                    self.pri, self.sec = self.sec, self.pri
                    return tmp

        def top(self):
            while True:
                tmp = self.pri.get() if not self.pri.empty() else None
                if tmp is not None: self.sec.put(tmp)
                if self.pri.empty():
                    self.pri, self.sec = self.sec, self.pri
                    return tmp

        def empty(self):
            return self.pri.empty()

'''796. Rotate String'''
    def rotateString(A, B):
        return not any([A, B]) or any(A[i:] + A[:i] == B for i in range(len(A)))

'''814. Binary Tree Pruning'''
    def pruneTree(root):
        def _pruneTree(node):
            if not node: return True
            l, r =  _pruneTree(node.left), _pruneTree(node.right)
            if l: node.left = None
            if r: node.right = None
            return not node.val and l and r
        _pruneTree(root)
        return root

'''797. All Paths From Source to Target'''
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

'''788. Rotated Digits'''
    def rotatedDigits(N):
        from collections import defaultdict
        dig_map = [0, 1, 5, -1, -1, 2, 9, -1, 8, 6]
        dp = [1, 1, 2, 0, 0, 2, 2, 0, 1, 2] #0: can't flip, 1: same, 2: diff
        for i in range(10, N + 1):
            last_dig = i % 10
            first_part = i / 10
            if dp[last_dig] == 2 and dp[first_part] or dp[first_part] == 2 and dp[last_dig]:
                dp.append(2)
            elif dp[last_dig] == dp[first_part] == 1:
                dp.append(1)
            else:
                dp.append(0)
        return sum(i == 2 for i in dp[1:N+1])

'''819. Most Common Word'''
    def mostCommonWord(paragraph, banned):
        from collections import Counter
        counts = Counter((''.join([char for char in paragraph if char.isalpha() or char == ' '])).lower().split(' '))
        freqs = sorted([(freq, word) for word, freq in counts.iteritems()], reverse=True)
        return next((word for freq, word in freqs if word not in banned), None)

'''844. Backspace String Compare'''
    def backspaceCompare(S, T):
        def process_str(string):
            res = []
            for c in string:
                if c != '#':
                    res.append(c)
                elif res:
                    res.pop()
            return res
        return process_str(S) == process_str(T)

'''784. Letter Case Permutation'''
    def letterCasePermutation(S):
        res = []
        S = list(S.lower())
        def _letterCasePermutation(i):
            if i == len(S):
                return res.append(''.join(S))
            elif S[i].isalpha():
                S[i] = S[i].upper()
                _letterCasePermutation(i + 1)
                S[i] = S[i].lower()
            _letterCasePermutation(i + 1)
        _letterCasePermutation(0)
        return res

'''171. Excel Sheet Column Number'''
    def titleToNumber(self, s):
        return reduce(lambda x, y: (ord(y) - ord('a') + 1) + (x * 26), s.lower(), 0)

'''783. Minimum Distance Between BST Nodes'''
    # iterative
    def minDiffInBST(root):
        # in order traversal
        cur = root
        stack = []
        def traverse_left(cur):
            while cur:
                stack.append(cur)
                cur = cur.left
        traverse_left(root)
        prev = None
        min_dif = float('inf')
        while stack:
            cur = stack[-1].val
            traverse_left(stack.pop().right)
            if prev is not None: min_dif = min(cur - prev, min_dif)
            prev = cur
        return min_dif

    # recursive
    def minDiffInBST(root):
        self.prev, self.min_dif = float('-inf'), float('inf')
        def in_order(node):
            if not node: return
            in_order(node.left)
            self.min_dif = min(self.min_dif, node.val - self.prev)
            self.prev = node.val
            in_order(node.right)
        in_order(root)
        return self.min_dif

'''789. Escape The Ghosts'''
    def escapeGhosts(ghosts, target):
        def dist(x, y): return sum(map(abs, [target[0] - x, target[1] - y]))
        dist_ghost = dist(0, 0)
        return not any(dist(x, y) <= dist_ghost for x, y in ghosts)

'''37. Sudoku Solver'''
    def solveSudoku(self, board):

        seen = set()

        def is_valid_add(i, j, el):
            seen_item = {(i, None, el), (None, j, el), (i/3, j/3, el)}
            if seen_item & seen: return False
            board[i][j] = el
            seen.update(seen_item)
            return True

        def remove_seen_item(i, j, el):
            for el in {(i, None, el), (None, j, el), (i/3, j/3, el)}:
                seen.remove(el)
            board[i][j] = '.'

        for i, col in enumerate(board):
            for j, el in enumerate(col):
                if el != '.': is_valid_add(i, j, el)  # add pre-existing

        def _solveSudoku(start_i):
            for i in range(start_i, 9 * 9):
                a, b = divmod(i, 9)
                if board[a][b] == '.':
                    for potential in map(str, range(1, 10)):
                        if is_valid_add(a, b, potential):
                            if _solveSudoku(i + 1): return True
                            remove_seen_item(a, b, potential)
                    return False
            return True
        _solveSudoku(0)

'''804. Unique Morse Code Words'''
    def uniqueMorseRepresentations(words):
        mapping = [".-","-...","-.-.","-..",".","..-.","--.",
                   "....","..",".---","-.-",".-..","--","-.",
                   "---",".--.","--.-",".-.","...","-","..-",
                   "...-",".--","-..-","-.--","--.."]
        def covert_to_morse(word):
            return ''.join(mapping[ord(char) - ord('a')] for char in word.lower())
        return len(set(map(covert_to_morse, words)))

'''771. Jewels and Stones'''
    def numJewelsInStones(J, S):
        jewel_set = set(list(J))
        return sum(stone in jewel_set for stone in S)

'''852. Peak Index in a Mountain Array'''
    def peakIndexInMountainArray(A):
        l, r = 0, len(A)
        while True:
            mid = (l + r) / 2
            if A[mid - 1] < A[mid] > A[mid + 1]:
                return mid
            elif A[mid - 1] < A[mid] < A[mid + 1]:
                l = mid
            else:
                r = mid

'''832. Flipping an Image'''
    def flipAndInvertImage(A):
        return [map(lambda x: 1 - x, reversed(lst)) for lst in A]

'''806. Number of Lines To Write String'''
    def numberOfLines(widths, S):
        num_lines = line_width = 0
        for i, char in enumerate(S):
            width = widths[ord(char) - ord('a')]
            if not num_lines or line_width + width > 100:
                line_width = width
                num_lines += 1
            else:
                line_width += width
        return (num_lines, line_width)

'''811. Subdomain Visit Count'''
    def subdomainVisits(cpdomains):
        from collections import defaultdict
        subdomain_counts = defaultdict(int)
        for cpdomain in cpdomains:
            count, domain = cpdomain.split(' ')
            domain = domain.split('.')
            for i in range(len(domain)):
                subdomain_counts['.'.join(domain[i:])] += int(count)
        return ["%d %s" %(count, domain) for domain, count in subdomain_counts.iteritems()]

'''821. Shortest Distance to a Character'''
    def shortestToChar(S, C):
        res = []
        last_C = float('-inf')
        for i, char in enumerate(S):
            if char == C:
                last_C = i
            res.append(i - last_C)
        last_C = float('inf')
        for i in range(len(S) -1, -1, -1):
            char = S[i]
            if char == C:
                last_C = i
            res[i] = min(res[i], last_C - i)
        return res

'''762. Prime Number of Set Bits in Binary Representation'''
    def countPrimeSetBits(L, R):
        primes = {2, 3, 5, 7, 11, 13, 17, 19}
        return sum((bin(i).count('1') in primes) for i in range(L, R + 1))

'''824. Goat Latin'''
    def toGoatLatin(S):
        def parse_prefix(word):
            return word if (word[0].lower() in 'aeiou') else (word[1:] + word[0])
        return ' '.join([word + 'ma' + ((i + 1) * 'a')
                         for i, word in enumerate(map(parse_prefix, S.split()))])

'''766. Toeplitz Matrix'''
    def isToeplitzMatrix(matrix):
        m, n = len(matrix), len(matrix[0])
        for i in range(m):
            for j in range(n):
                if i and j and matrix[i - 1][j - 1] != matrix[i][j]:
                    return False
        return True

'''830. Positions of Large Groups'''
    def largeGroupPositions(S):
        stack = []
        prev = None
        for i, cur in enumerate(S):
            if cur != prev: stack.append(i)
            prev = cur
        stack.append(len(S))
        return [(stack[i - 1], stack[i] - 1) for i in range(1, len(stack)) if stack[i] - stack[i-1] >= 3]

'''746. Min Cost Climbing Stairs'''
    def minCostClimbingStairs(cost):
        if len(cost) <= 2: return int(bool(len(cost)))
        dp_cost_step = cost[:2]
        for i in range(2, len(cost)):
            dp_cost_step.append(min(dp_cost_step[-2:]) + cost[i])
        return min(dp_cost_step[-2:])

'''747. Largest Number At Least Twice of Others'''
    def dominantIndex(nums):
        max_num, max_idx = max((val, idx) for idx, val in enumerate(nums))
        return max_idx if all(max_num >= num * 2 for num in nums if num != max_num) else -1

'''744. Find Smallest Letter Greater Than Target'''
    def nextGreatestLetter(letters, target):
        return next((char for char in letters if target < char), letters[0])

'''836. Rectangle Overlap'''
    def isRectangleOverlap(rec1, rec2):
        x1, y1 = max(rec1[0], rec2[0]), max(rec1[1], rec2[1])
        x2, y2 = min(rec1[2], rec2[2]), min(rec1[3], rec2[3])
        return x1 < x2 and y1 < y2

'''482. License Key Formatting'''
    def licenseKeyFormatting(S, K):
        S = S.replace('-', '').upper()
        return '-'.join([ S[max(len(S) - i, 0) : len(S) - i + K]
                         for i in range(K, len(S) + K, K)][::-1])

'''849. Maximize Distance to Closest Person'''
    def maxDistToClosest(seats):
        max_dist = prev = 0
        for i in range(1, len(seats)):
            if not seats[i - 1] and seats[i]:
                max_dist = max(max_dist, i - prev)
            elif seats[i - 1] and not seats[i]:
                prev = i - 1
        return max(max_dist / 2,  # max gap
                   next((i for i, seated in enumerate(seats) if seated == 1)),  # from left
                   next((i for i in range(len(seats)) if seats[len(seats) - i - 1]))  # from right
                  )

'''840. Magic Squares In Grid'''
    def numMagicSquaresInside(grid):
        def is_magic(i, j):
            qwe, asd, zxc = [grid[i + k][j:j + 3] for k in range(3)]
            (q,w,e), (a,s,d), (z,x,c) = qwe, asd, zxc
            qaz, wsx, edc = (q,a,z),(w,s,x),(e,d,c)
            qsc, esz = (q,s,c),(e,s,z)
            totals = map(sum,[qwe, asd, zxc, qaz, wsx, edc, qsc, esz])
            return {q,w,e,a,s,d,z,x,c} == {1,2,3,4,5,6,7,8,9} and \
                    all(totals[i] == totals[i - 1] for i in range(1, len(totals)))
        
        return sum(is_magic(i, j) for i in range(len(grid) - 2) for j in range(len(grid[0]) - 2))

'''405. Convert a Number to Hexadecimal'''
    def toHex(num):
        if not num: return '0'
        res = ''
        mask = reduce(lambda x,y: x | (1 << y), range(4), 0)
        mapping = '0123456789abcdef'
        for _ in range(8):
            res = mapping[(num & mask) % len(mapping)] + res
            num >>= 4
        return res.lstrip('0')

'''807. Max Increase to Keep City Skyline'''

    def maxIncreaseKeepingSkyline(self, grid):
        m, n = len(grid), len(grid[0])
        hor_view, ver_view = [0] * m, [0] * n
        for i in range(m):
            for j in range(n):
                hor_view[i] = max(hor_view[i], grid[i][j])
                ver_view[j] = max(ver_view[j], grid[i][j])
        total = sum(
            min(hor_view[i], ver_view[j]) - grid[i][j]
            for i in range(m) for j in range(n)
        )
        return total

'''763. Partition Labels'''
    def partitionLabels(S):
        from collections import Counter
        counts = Counter(S)
        i = j = 0
        res = []
        while i < len(S):
            seen = set([S[j]])
            while j < len(S) and seen:
                char = S[j]
                seen.add(char)
                counts[char] -= 1
                if not counts[char]: seen.remove(char)
                j += 1
            res.append(j - i)
            i = j
        return res

'''791. Custom Sort String'''
    def customSortString(S, T):
        from collections import Counter
        counts = Counter(T)
        seq = S + ''.join(set(T)-set(S))
        return ''.join(char * counts[char] for char in seq)

'''841. Keys and Rooms'''
    def canVisitAllRooms(rooms):
        if not rooms: return True
        visited, bfs = {0}, {0}
        while bfs:
            bfs = {next_room
                   for room in bfs
                   for next_room in rooms[room]
                   if next_room not in visited and (visited.add(next_room) is None)}
        return len(visited) == len(rooms)

'''553. Optimal Division'''
    def optimalDivision(nums):
        return '/'.join(map(str, nums)) if len(nums) <= 2 else \
            '%d/(%s)' %(nums[0], '/'.join(map(str, nums[1:])))

'''739. Daily Temperatures'''
    def dailyTemperatures(temperatures):
        res, stack = [], []
        for i in range(len(temperatures) - 1, -1, -1):
            while stack and temperatures[stack[-1]] <= temperatures[i]:
                stack.pop()
            res.append(stack[-1] - i if stack else 0)
            stack.append(i)
        res.reverse()
        return res

'''748. Shortest Completing Word'''
    def shortestCompletingWord(licensePlate, words):
        from collections import Counter
        counts = Counter(char.lower() for char in licensePlate if char.isalpha())
        shortest_word = shortest_len = None
        for word in words:
            word_char_counts = Counter(word.lower())
            if (shortest_len is None or len(word) < shortest_len) and \
                all(count <= word_char_counts[plate_c] for plate_c, count in counts.iteritems()):
                shortest_word, shortest_len = word, len(word)
        return shortest_word

'''817. Linked List Components'''
    def numComponents(head, G):
        G = set(G)
        cur = head
        prev_in_G = False
        total = 0
        while cur:
            if not prev_in_G and cur.val in G:
                total += 1
            prev_in_G = cur.val in G
            cur = cur.next
        return total

'''565. Array Nesting'''
    def arrayNesting(nums):
        max_dep = 0
        for i in range(len(nums)):
            cur, depth = i, 0
            while nums[cur] is not None:
                nums[cur], cur = None, nums[cur]
                depth += 1
            max_dep = max(depth, max_dep)
        return max_dep

'''781. Rabbits in Forest'''
    def numRabbits(answers):
        from collections import Counter
        from math import ceil
        counts = Counter(x + 1 for x in answers)
        return int(sum(ceil(float(head_count) / quantity) * quantity
                   for quantity, head_count in counts.iteritems()))

'''421. Maximum XOR of Two Numbers in an Array'''
    def findMaximumXOR(nums):
        class Tree(object):
            class Node(object):
                def __init__(self, left=None, right=None):
                    self.o_l = [left, right]

            def __init__(self):
                self.node_to_vals = {}
                self.root = self.Node()

            def insert(self, num):
                cur = self.root
                for bin_dig in reversed([bool((1 << i) & num) for i in range(32)]):
                    if not cur.o_l[bin_dig]: cur.o_l[bin_dig] = self.Node()
                    cur = cur.o_l[bin_dig]
                self.node_to_vals[cur] = num

            def find_max_xor(self, num):
                cur = self.root
                for bin_dig in reversed([bool((1 << i) & num) for i in range(32)]):
                    next_node = cur.o_l[not bin_dig] or cur.o_l[bin_dig]
                    cur = next_node
                return self.node_to_vals[cur]
        tree = Tree()
        for num in nums: tree.insert(num)
        return max(num ^ tree.find_max_xor(num) for num in nums)

    def findMaximumXOR(nums):
        res = 0
        for i in reversed(range(32)):
            prefixes = set(x >> i for x in nums)
            res <<= 1
            res = res | any((res|1) ^ p in prefixes for p in prefixes)
        return res

'''6. ZigZag Conversion'''
    def convert(s, numRows):
        res = [[] for _ in range(numRows)]
        cur, direction = 0, 1
        for i, char in enumerate(s):
            res[cur].append(char)
            # change direction if the next idx is out of range
            direction *= 1 if (0 <= (direction + cur) < numRows) else -1
            cur += direction
        return ''.join([''.join(row) for row in res])

'''79. Word Search'''
    def exist(board, word):
        m, n = len(board), len(board[0])
        visited = [[False] * n for _ in range(m)]
        def exist(i, j, char_i):
            if word[char_i] != board[i][j]: return False
            elif char_i == len(word) - 1 and not visited[i][j]: return True
            adjs = [(i + a, j + b)
                    for a, b in zip([1,0,-1,0], [0,1,0,-1])
                    if (0 <= i + a < m) and (0 <= j + b < n)]
            visited[i][j] = True
            if any(not visited[x][y] and exist(x, y, char_i + 1)
                   for x, y in adjs):
                    return True
            visited[i][j] = False
            return False

        return any(exist(i, j, 0)
                   for i in range(m)
                   for j in range(n))

'''130. Surrounded Regions'''
    def solve(board):
        if not board: return
        m, n = len(board), len(board[0])
        bfs = [pair for x in range(m) for pair in [(x, 0),(x, n - 1)]] + \
              [pair for y in range(n) for pair in [(0, y),(m - 1, y)]]
        
        while bfs:
            x, y = bfs.pop()
            if 0 <= x < m and 0 <= y < n and board[x][y] == 'O':
                board[x][y] = ''
                bfs.extend([x + a, y + b] for a, b in zip([0,1,0,-1], [1,0,-1,0]))

        for i in range(m):
            for j in range(n):
                board[i][j] = 'X' if board[i][j] else 'O'

'''820. Short Encoding of Words'''
    def minimumLengthEncoding(words):
        words = sorted([word[::-1] for word in words], reverse=True)
        return sum(len(words[i]) + 1
                   for i in range(len(words) -1, -1, -1)
                   if not (i - 1 >= 0 and words[i - 1].startswith(words[i])))

    def minimumLengthEncoding(words):
        words = [word[::-1] for word in words]
        tree = {}
        for word in words:
            cur = tree
            for char in word:
                cur = cur.setdefault(char, {})
        self.res = 0
        def dfs(depth, cur):
            if not cur:
                self.res += depth + 1
                return
            for adj in cur:
                dfs(depth + 1, cur[adj])
        dfs(0, tree)
        return self.res

'''638. Shopping Offers'''
    def shoppingOffers(price, special, needs):
        def dfs(total=0, idx=0):
            if idx == len(special):
                return total + sum(need * price[i] for i, need in enumerate(needs))
            cost = []
            n, quan = len(needs), 0
            while all(needs[i] - quan * special[idx][i] >= 0 for i in range(n)):
                for i in range(n): needs[i] = needs[i] - quan * special[idx][i]
                cost.append(dfs(total + special[idx][-1] * quan, idx + 1))
                for i in range(n): needs[i] = needs[i] + quan * special[idx][i]
                quan += 1
            return min(c for c in cost)
        return dfs()

'''848. Shifting Letters'''
    def shiftingLetters(S, shifts):
        cur = 0
        for i in range(len(shifts) -1, -1, -1):
            shifts[i] += cur
            cur = shifts[i]
        return ''.join(chr((ord(S[i]) - ord('a') + shifts[i]) % 26 + ord('a'))
                       for i in range(len(shifts)))

'''767. Reorganize String'''
    def reorganizeString(S):
        from collections import Counter
        counts = [(count, char) for char, count in Counter(S).iteritems()]
        max_freq, max_freq_char = max(counts)
        if max_freq > ((len(S) + 1)/ 2): return ""
        
        res = [[max_freq_char] for _ in range(max_freq)]
        i = 0
        while counts:
            count, char = counts.pop()
            if char != max_freq_char:
                for j in range(i, i + count):
                    res[j % max_freq].append(char)
                i += count
        return ''.join([''.join(x) for x in res])

'''868. Transpose Matrix'''
    def transpose(A):
        return [[A[i][j] for i in range(len(A))]
                for j in range(len(A[0]))] or [[]]

'''866. Smallest Subtree with all the Deepest Nodes'''
    def subtreeWithAllDeepest(root):
        bfs = {root}
        while bfs:
            prev = bfs
            bfs = {kid for node in bfs for kid in [node.left, node.right] if kid}
        deepest = prev
        def dfs_deepest(node):
            if not node or node in deepest: return node
            l, r = dfs_deepest(node.left), dfs_deepest(node.right)
            return node if l and r else l or r
        return dfs_deepest(root)

'''306. Additive Number'''
    def isAdditiveNumber(num):
        def is_seq(i, j, k):
            if k == len(num): return True
            a, b = int(num[i:j]), int(num[j:k])
            if len(str(a)) != j - i or len(str(b)) != k - j: return False
            total = str(a + b)
            return num[k:].startswith(total) and is_seq(j, k, k + len(total))
        return any(is_seq(0, i, j)
                   for j in range(2, len(num))
                   for i in range(1, j))

'''888. Fair Candy Swap'''
    def fairCandySwap(A, B):
        n, m = sum(A), sum(B)
        gap = (m - n) / 2
        B_set = set(B)
        for i in A:
            if (i + gap) in B_set:
                return (i, (i + gap))

'''908. Smallest Range I'''
    def smallestRangeI(A, K):
        return max(max(A) - min(A) - 2 * K, 0)

'''911. Online Election'''
import bisect
class TopVotedCandidate(object):

    def __init__(self, persons, times):        
        person_to_count = {}
        max_vote = 0
        winning = self.winning = []
        self.times = times
        for i in range(len(persons)):
            person_to_count[persons[i]] = person_to_count.get(persons[i], 0) + 1
            if person_to_count[persons[i]] >= max_vote:
                max_vote = person_to_count[persons[i]]
                winning.append(persons[i])
            else:
                winning.append(winning[-1])            

    def q(self, t):
        idx = bisect.bisect_left(self.times, t)
        if idx >= len(self.times) or self.times[idx] > t: idx -= 1
        return self.winning[idx]

'''896. Monotonic Array'''
    def isMonotonic(A):
        if len(A) <= 2: return True
        return all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or \
               all(A[i] >= A[i + 1] for i in range(len(A) - 1))

'''897. Increasing Order Search Tree'''
class Solution(object):
    def increasingBST(self, root):
        self.cur = dummy = TreeNode('dummy')
        def create_tree(node):
            if not node: return
            if node.left:
                create_tree(node.left)
            self.cur.right = TreeNode(node.val)
            self.cur = self.cur.right
            if node.right:
                create_tree(node.right)
        create_tree(root)
        return dummy.right

'''884. Uncommon Words from Two Sentences'''
    def uncommonFromSentences(A, B):
        from collections import Counter
        A = A.split(' ')
        B = B.split(' ')
        A = Counter(A)
        B = Counter(B)
        return [a for a, count in A.iteritems() if count == 1 and a not in B] + \
            [b for b, count in B.iteritems() if count == 1 and b not in A]

'''892. Surface Area of 3D Shapes'''
    def surfaceArea(grid):
        n = len(grid)
        def surface(n): return n * 6 - (n - 1) * 2 if n else 0
        total = sum(surface(grid[i][j]) for i in range(n) for j in range(n))
        adj_ver = sum(min(grid[i][j], grid[i][j + 1]) * 2 for i in range(n) for j in range(n - 1))
        adj_hor = sum(min(grid[i][j], grid[i + 1][j ]) * 2 for i in range(n - 1) for j in range(n))
        return total - adj_ver - adj_hor

'''894. All Possible Full Binary Trees'''
    def allPossibleFBT(self, N):
        def construct(n):
            if n == 1: return [TreeNode(0)]
            trees = []
            for k in range(1, n, 2):
                left, right = construct(k), construct(n - k - 1)
                for l in left:
                    for r in right:
                        node = TreeNode(0)
                        node.left, node.right = l, r
                        trees.append(node)
            return trees
        return construct(N)

'''914. X of a Kind in a Deck of Cards'''
    def hasGroupsSizeX( deck):
        from collections import Counter
        counts = Counter(deck)
        counts = set(counts.values())
        min_c = min(counts)
        return any( all((c % i == 0) for c in counts) for i in range(2, min_c + 1)) and min_c > 1

'''915. Partition Array into Disjoint Intervals'''
    def partitionDisjoint(self, A):
        r_min = [A[-1]]
        for i in range(len(A) -2, -1, -1):
            r_min.append(min(A[i], r_min[-1]))
        r_min.reverse()

        max_so_far = A[0]
        for i in range(len(A) - 1):
            if max_so_far <=  r_min[i + 1]:
                return i + 1
            max_so_far = max(max_so_far, A[i])
        return len(r_min)

'''916. Word Subsets'''
    def wordSubsets(A, B):
        from collections import Counter        
        max_c = {}
        for w2 in B:
            for char, count in Counter(w2).iteritems():
                max_c[char] = max(max_c.get(char, 0), count)

        res = []
        for w1 in A:
            c_word1 = Counter(w1)
            if all(char in c_word1 and c_word1[char] >= max_c[char] for char in max_c):
                res.append(w1)
        return res

'''925. Long Pressed Name'''
    def isLongPressedName(name, typed):
        def is_long_pressed(name, typed):
            if not name and not typed:
                return True
            elif not name or not typed:
                return False
            elif name[0] != typed[0]:
                return False
            else:
                n_dif, t_dif = len(name), len(typed)
                new_name, new_typed = name.lstrip(name[0]), typed.lstrip(typed[0])
                n_dif -= len(new_name)
                t_dif -= len(new_typed)
                return n_dif <= t_dif and is_long_pressed(new_name, new_typed)
        return is_long_pressed(name, typed)

'''926. Flip String to Monotone Increasing'''
    def minFlipsMonoIncr(S):
        if not S:
            return 0
        num_zeros_right = []
        zeros_so_far = 0
        count = 0
        for bit in reversed(S):
            count += 1
            zeros_so_far += (bit == '0')
            num_zeros_right.append(zeros_so_far)
        num_zeros_right = num_zeros_right[::-1]
        num_zeros_right.append(0)

        optimal = S.count('0')
        ones_so_far = 0
        for i, bit in enumerate(S):
            ones_so_far += (bit == '1')
            zeros_to_right = num_zeros_right[i + 1]
            optimal = min(optimal, ones_so_far + zeros_to_right)
        return optimal

'''929. Unique Email Addresses'''
    def numUniqueEmails(emails):
        res = set()
        for email in emails:
            email = email.split("@")
            email[0] = email[0].replace(".", "")
            email[0] = email[0][:email[0].find("+")]
            res.add(tuple(email))
        return len(res)

'''930. Binary Subarrays With Sum'''
    def numSubarraysWithSum(A, S):
        sum_to_idx = [A[0]]
        for i in range(1, len(A)):
            sum_to_idx.append(sum_to_idx[-1] + A[i])
        res = sum_so_far = 0
        seen_count = {}
        for num in A:
            seen_count[sum_so_far] = seen_count.get(sum_so_far, 0) + 1
            res += seen_count.get((sum_so_far + num) - S, 0)
            sum_so_far += num
        return res

'''931. Minimum Falling Path Sum'''
    def minFallingPathSum(A):
        prev = A[0][:]
        row = []
        n = len(A)
        for i in range(1, n):
            row = []
            for j in range(n):
                cur = min((prev[j - 1] if (j - 1) >= 0 else float('inf')),
                          (prev[j + 1]  if (j + 1) < n else float('inf')),
                          prev[j])
                row.append(cur + A[i][j])
            prev = row
        return min(row or prev)
