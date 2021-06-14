ls = {LIST}
{result} = ls[0]
for num in ls[1::]:
    {result} = math.gcd({result}, num)
