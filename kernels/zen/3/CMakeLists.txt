##Copyright (C) 2020-2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_gemm_small.c
        bli_trsm_small.c
        bli_dgemm_ref_k1.c
        bli_zgemm_ref_k1.c
    )

add_subdirectory(sup)
