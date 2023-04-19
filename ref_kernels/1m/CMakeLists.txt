##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
        PRIVATE
    bli_packm_cxk_1er_ref.c
    bli_packm_cxk_3mis_ref.c
    bli_packm_cxk_4mi_ref.c
    bli_packm_cxk_bb_ref.c
    bli_packm_cxk_ref.c
    bli_packm_cxk_rih_ref.c
    bli_unpackm_cxk_ref.c
    )
