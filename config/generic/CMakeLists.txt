##Copyright (C) 2021, Advanced Micro Devices, Inc. All rights reserved. ##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_cntx_init_generic.c
        bli_family_generic.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
