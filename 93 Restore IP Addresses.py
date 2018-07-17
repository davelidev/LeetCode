class Answer(object):
'''93. Restore IP Addresses'''
    def restoreIpAddresses(s):
        n = len(s)
        if float(n) / 4 > 3: return []
        dp = [[] for _ in range(n + 1)]
        dp[0].append([])
        for j in range(1, n + 1):
            for i in range(max(j - 3, 0), j):
                for ip in dp[i]:
                    if ((s[i:j] == '0') or                         (not s[i:j].startswith('0') and int(s[i:j]) < 256))                          and len(ip) < 4:
                        dp[j].append(ip + [s[i:j]])

        return ['.'.join(ip) for ip in dp[n] if len(ip) == 4]