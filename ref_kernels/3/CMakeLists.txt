##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
        PRIVATE
    bli_gemm_ref.c
    bli_gemmsup_ref.c
    bli_gemmtrsm_ref.c
    bli_trsm_ref.c
    )

add_subdirectory(bb)
