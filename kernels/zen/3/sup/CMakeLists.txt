##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_gemmsup_rd_zen_asm_s6x16.c
        bli_gemmsup_rd_zen_asm_s6x16m.c
        bli_gemmsup_rd_zen_asm_s6x16n.c
        bli_gemmsup_rv_zen_asm_c3x8.c
        bli_gemmsup_rv_zen_asm_c3x8m.c
        bli_gemmsup_rv_zen_asm_c3x8n.c
        bli_gemmsup_rv_zen_asm_s6x16.c
        bli_gemmsup_rv_zen_asm_s6x16m.c
        bli_gemmsup_rv_zen_asm_s6x16n.c
        bli_gemmsup_rv_zen_asm_z3x4.c
        bli_gemmsup_rv_zen_asm_z3x4m.c
        bli_gemmsup_rv_zen_asm_z3x4n.c
    )
