##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_gemmtrsm_l_zen_16x14.c
        bli_gemmtrsm_u_zen_16x14.c
    )
