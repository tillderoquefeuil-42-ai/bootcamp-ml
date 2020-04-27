from TinyStatistician import TinyStatistician 

tstat = TinyStatistician()

a = [1, 42, 300, 10, 59]
a = [1, 10, 42, 59, 300]

print(tstat.mean(a))
# 82,4
print(tstat.median(a))
# 42.0
print(tstat.quartiles(a, 25))
# 10.0
print(tstat.quartiles(a, 75))
# 59.0
print(tstat.var(a))
# 12279.439999999999
print(tstat.std(a))
# 110.81263465868862