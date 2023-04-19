##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_addv_ref.c
        bli_amaxv_ref.c
        bli_aminv_ref.c
        bli_axpbyv_ref.c
        bli_axpyv_ref.c
        bli_copyv_ref.c
        bli_dotv_ref.c
        bli_dotxv_ref.c
        bli_invertv_ref.c
        bli_scal2v_ref.c
        bli_scalv_ref.c
        bli_setv_ref.c
        bli_subv_ref.c
        bli_swapv_ref.c
        bli_xpbyv_ref.c
    )
