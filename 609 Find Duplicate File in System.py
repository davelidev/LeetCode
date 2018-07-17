class Answer(object):
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