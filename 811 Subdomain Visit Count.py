class Answer(object):'''811. Subdomain Visit Count'''
    def subdomainVisits(cpdomains):
        from collections import defaultdict
        subdomain_counts = defaultdict(int)
        for cpdomain in cpdomains:
            count, domain = cpdomain.split(' ')
            domain = domain.split('.')
            for i in range(len(domain)):
                subdomain_counts['.'.join(domain[i:])] += int(count)
        return ["%d %s" %(count, domain) for domain, count in subdomain_counts.iteritems()]