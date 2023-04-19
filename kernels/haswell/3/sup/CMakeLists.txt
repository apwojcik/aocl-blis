##Copyright (C) 2020-2021, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_gemmsup_rd_haswell_asm_d6x8m.c
        bli_gemmsup_rd_haswell_asm_d6x8n.c
        bli_gemmsup_rv_haswell_asm_d6x8m.c
        bli_gemmsup_rv_haswell_asm_d6x8n.c
    )

add_subdirectory(d6x8)


