##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_unpackm.h
        bli_unpackm_blk_var1.c
        bli_unpackm_blk_var1.h
        bli_unpackm_check.c
        bli_unpackm_check.h
        bli_unpackm_cntl.c
        bli_unpackm_cntl.h
        bli_unpackm_cxk.c
        bli_unpackm_cxk.h
        bli_unpackm_int.c
        bli_unpackm_int.h
        bli_unpackm_unb_var1.c
        bli_unpackm_unb_var1.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
