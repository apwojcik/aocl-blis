##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved. ##

#Add source files to target
target_sources(${TARGET_NAME}
    PRIVATE
        bli_cntx_init_haswell.c
        bli_family_haswell.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )