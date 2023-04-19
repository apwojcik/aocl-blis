##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_gemmt.h
        bli_gemmt_front.c
        bli_gemmt_front.h
        bli_gemmt_ker_var2.c
        bli_gemmt_sup_var1n2m.c
        bli_gemmt_var.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
