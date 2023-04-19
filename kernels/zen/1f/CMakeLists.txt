##Copyright (C) 2020-2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_axpyf_zen_int_8.c
        bli_dotxf_zen_int_8.c
        bli_axpyf_zen_int_5.c
        bli_axpyf_zen_int_4.c
        bli_axpyf_zen_int_6.c
        bli_axpy2v_zen_int.c
        bli_dotxaxpyf_zen_int_8.c
    )
