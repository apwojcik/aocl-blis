##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved. ##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_cntx_init_zen2.c
        bli_family_zen2.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
