##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_gemm1m_ref.c
        bli_gemm3m1_ref.c
        bli_gemm3mh_ref.c
        bli_gemm4m1_ref.c
        bli_gemm4mb_ref.c
        bli_gemm4mh_ref.c
        bli_gemmtrsm1m_ref.c
        bli_gemmtrsm3m1_ref.c
        bli_gemmtrsm4m1_ref.c
        bli_trsm1m_ref.c
        bli_trsm3m1_ref.c
        bli_trsm4m1_ref.c
    )
