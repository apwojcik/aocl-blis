##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_cntx_ind_stage.c
        bli_cntx_ind_stage.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )