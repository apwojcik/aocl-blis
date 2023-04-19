##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_amaxv_zen_int.c
        bli_axpbyv_zen_int.c
        bli_axpbyv_zen_int10.c
        bli_axpyv_zen_int.c
        bli_axpyv_zen_int10.c
        bli_copyv_zen_int.c
        bli_dotv_zen_int.c
        bli_dotv_zen_int10.c
        bli_dotxv_zen_int.c
        bli_scalv_zen_int.c
        bli_scalv_zen_int10.c
        bli_setv_zen_int.c
        bli_swapv_zen_int8.c
        bli_norm2_zen_int.c
    )

target_compile_options(${TARGET_NAME} PRIVATE -mfma)