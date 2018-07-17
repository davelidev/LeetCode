class Answer(object):
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