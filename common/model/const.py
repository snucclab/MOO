# Encoder configurations
MDL_ENCODER = 'encoder'
DEF_ENCODER = 'google/electra-base-discriminator'

# Decoder configurations
MDL_DECODER = 'decoder'
MDL_D_PATH = 'path'
MDL_D_ENC = 'encoder_config'
MDL_D_EMBED = 'embedding_dim'
MDL_D_HIDDEN = 'hidden_dim'
MDL_D_INTER = 'intermediate_dim'
MDL_D_LAYER = 'layer'
MDL_D_INIT = 'init_factor'
MDL_D_LN_EPS = 'layernorm_eps'
MDL_D_HEAD = 'head'

# Decoder configuration default
DEF_D_EMBED = 128
DEF_D_HIDDEN = 768
DEF_D_INTER = 2048
DEF_D_LAYER = 6
DEF_D_INIT = 0.02
DEF_D_LN_EPS = 1E-8
DEF_D_HEAD = 12

# Index for padding
PAD_ID = -1

# Infinity values (we use 1E10 for numerical stability)
NEG_INF = float('-inf')
NEG_INF_SAFE = -1E10
POS_INF = float('inf')
POS_INF_SAFE = 1E10
FLOAT_NAN = float('NaN')

# Maximum value of index
TXT_MAX = 500
NUM_MAX = TXT_MAX
RES_MAX = 128
