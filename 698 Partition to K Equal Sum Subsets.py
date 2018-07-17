class Answer(object):
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