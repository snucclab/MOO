place = 10 ** ({place} - 1)
digit = ({original} // place) % 10
{result} = {original} + ({change} - digit) * place