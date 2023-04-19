##Copyright (C) 2020-2021, Advanced Micro Devices, Inc. All rights reserved.## 

target_sources(${TARGET_NAME}
    PRIVATE
        bli_gemmsup_r_haswell_ref_sMx1.c
        bli_gemmsup_rd_haswell_asm_sMx1.c
        bli_gemmsup_rd_haswell_asm_sMx12.c
        bli_gemmsup_rd_haswell_asm_sMx16.c
        bli_gemmsup_rd_haswell_asm_sMx2.c
        bli_gemmsup_rd_haswell_asm_sMx4.c
        bli_gemmsup_rd_haswell_asm_sMx8.c
        bli_gemmsup_rv_haswell_asm_sMx12.c
        bli_gemmsup_rv_haswell_asm_sMx16.c
        bli_gemmsup_rv_haswell_asm_sMx2.c
        bli_gemmsup_rv_haswell_asm_sMx4.c
        bli_gemmsup_rv_haswell_asm_sMx6.c
        bli_gemmsup_rv_haswell_asm_sMx8.c
    )
