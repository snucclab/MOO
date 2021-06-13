from math import gcd
ls = {LIST}
result = ls[0]
for num in ls[1::]:
    result = gcd(result, num)
