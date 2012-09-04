################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= sugarscape

# Cuda source files (compiled with cudacc)
CUFILES_sm_20	:= sugarscape.cu randoms.cu count.cu move.cu die.cu exercise_locks.cu \
	harvest.cu eat.cu age.cu grow_back1.cu mate.cu

USEGLLIB	:= 1
USEPARAMGL	:= 1
USEGLUT		:= 1
USECURAND   := 1

################################################################################
# Rules and targets

include ../../common/common.mk
