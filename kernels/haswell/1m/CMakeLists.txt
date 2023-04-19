##Copyright (C) 2020-2021, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_packm_haswell_asm_c3xk.c
        bli_packm_haswell_asm_c8xk.c
        bli_packm_haswell_asm_d6xk.c
        bli_packm_haswell_asm_d8xk.c
        bli_packm_haswell_asm_s16xk.c
        bli_packm_haswell_asm_s6xk.c
        bli_packm_haswell_asm_z3xk.c
        bli_packm_haswell_asm_z4xk.c
    )
