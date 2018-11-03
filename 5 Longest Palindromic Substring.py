class Answer(object):'''5. Longest Palindromic Substring'''
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