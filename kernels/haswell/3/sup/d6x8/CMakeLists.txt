##Copyright (C) 2020-2021, Advanced Micro Devices, Inc. All rights reserved.## 

target_sources(${TARGET_NAME}
    PRIVATE
        bli_gemmsup_r_haswell_ref_dMx1.c
        bli_gemmsup_rd_haswell_asm_dMx1.c
        bli_gemmsup_rd_haswell_asm_dMx2.c
        bli_gemmsup_rd_haswell_asm_dMx4.c
        bli_gemmsup_rd_haswell_asm_dMx8.c
        bli_gemmsup_rv_haswell_asm_dMx2.c
        bli_gemmsup_rv_haswell_asm_dMx4.c
        bli_gemmsup_rv_haswell_asm_dMx6.c
        bli_gemmsup_rv_haswell_asm_dMx8.c
    )
