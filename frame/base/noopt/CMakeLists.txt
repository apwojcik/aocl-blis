##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_dlamch.c
        bli_dlamch.h
        bli_lsame.c
        bli_lsame.h
        bli_slamch.c
        bli_slamch.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
