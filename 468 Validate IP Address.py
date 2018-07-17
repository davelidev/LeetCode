class Answer(object):
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