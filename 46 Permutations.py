class Answer(object):'''46. Permutations'''
    def permute(self, nums):
        res = [[]]
        for num in nums:
            new_res =[]
            for item in res:
                for i in range(len(item) + 1):
                    new_res.append(item[:i] + [num] + item[i:])
            res = new_res
        return res