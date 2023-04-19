##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_dgemm_skx_asm_16x14.c
        bli_dgemm_skx_asm_16x12_l2.c
        bli_sgemm_skx_asm_32x12_l2.c
    )
