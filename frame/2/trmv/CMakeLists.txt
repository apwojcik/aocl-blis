##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_trmv.h
        bli_trmv_unb_var1.c
        bli_trmv_unb_var2.c
        bli_trmv_unf_var1.c
        bli_trmv_unf_var2.c
        bli_trmv_var.h
        bli_trmv_var_oapi.c
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
