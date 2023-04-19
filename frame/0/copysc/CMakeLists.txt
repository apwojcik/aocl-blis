##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved. ##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_copysc.c
        bli_copysc.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
