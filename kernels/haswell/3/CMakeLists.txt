##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_gemm_haswell_asm_d6x8.c
        bli_gemm_haswell_asm_d8x6.c
        bli_gemmtrsm_l_haswell_asm_d6x8.c
        bli_gemmtrsm_u_haswell_asm_d6x8.c
    )

add_subdirectory(sup)
