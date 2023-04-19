##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_trsm.h
        bli_trsm_blk_var1.c
        bli_trsm_blk_var2.c
        bli_trsm_blk_var3.c
        bli_trsm_cntl.c
        bli_trsm_cntl.h
        bli_trsm_front.c
        bli_trsm_front.h
        bli_trsm_int.c
        bli_trsm_int.h
        bli_trsm_ll_ker_var2.c
        bli_trsm_lu_ker_var2.c
        bli_trsm_packab.c
        bli_trsm_rl_ker_var2.c
        bli_trsm_ru_ker_var2.c
        bli_trsm_var.h
        bli_trsm_xx_ker_var2.c
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
