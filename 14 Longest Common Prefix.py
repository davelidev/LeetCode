class Answer(object):'''14. Longest Common Prefix'''
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