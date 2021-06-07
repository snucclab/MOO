import re

PROPERNOUN_PATTERN = re.compile('(\\([가나다라마바사아자차카타파하]\\)|정국|지민|석진|태형|남준|윤기|호석|민영|유정|은지|유나)')
VARIABLE_PATTERN = re.compile('([A-Z])')
NUMBER_PATTERN = re.compile('((?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?)')
OPERATOR_PATTERN = re.compile('(-|\\+|\\*+|/+|%|=|\\^)')

PROPER_BEGIN_PATTERN = re.compile('^' + PROPERNOUN_PATTERN.pattern)
VARIABLE_BEGIN_PATTERN = re.compile('^' + VARIABLE_PATTERN.pattern)
NUMBER_BEGIN_PATTERN = re.compile('^' + NUMBER_PATTERN.pattern)
