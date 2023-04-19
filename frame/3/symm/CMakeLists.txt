##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_symm.h
        bli_symm_front.c
        bli_symm_front.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
