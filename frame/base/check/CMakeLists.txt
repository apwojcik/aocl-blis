##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_obj_check.c
        bli_obj_check.h
        bli_part_check.c
        bli_part_check.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
