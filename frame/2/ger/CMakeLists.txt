##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_ger.h
        bli_ger_unb_var1.c
        bli_ger_unb_var2.c
        bli_ger_var_oapi.c
        bli_ger_var.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
