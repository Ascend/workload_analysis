# !/usr/bin/env bash
ASCEND_HOME=/home/hyp/ascend-toolkit/latest
ASCEND_DRIVER_HOME=/usr/local/Ascend
SCRIPT=$(readlink -f \"$0\")
SHELL_FOLDER=$(dirname \"$SCRIPT\")
export LD_LIBRARY_PATH=${ASCEND_DRIVER_HOME}/driver/lib64:${ASCEND_DRIVER_HOME}/driver/lib64/common:${ASCEND_DRIVER_HOME}/driver/lib64/driver:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ASCEND_HOME}/lib64:${ASCEND_HOME}/compiler/lib64/plugin/opskernel:${ASCEND_HOME}/compiler/lib64/plugin/nnengine:$LD_LIBRARY_PATH
export PYTHONPATH=${ASCEND_HOME}/python/site-package:${ASCEND_HOME}/opp/op_impl/built-in/ai_core/tbe:$PYTHONPATH
export PYTHONPATH=${SHELL_FOLDER}/op_modeling:$PYTHONPATH
export PATH=${ASCEND_HOME}/tools/profiler/bin:${ASCEND_HOME}/bin:${ASCEND_HOME}/compiler/ccec_complier/bin:$PATH
export ASCEND_AICPU_PATH=${ASCEND_HOME}
export ASCEND_OPP_PATH=${ASCEND_HOME}/opp
export TOOLCHAIN_HOME=${ASCEND_HOME}/toolkit
export ASCEND_HOME_PATH=${ASCEND_HOME}
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=1