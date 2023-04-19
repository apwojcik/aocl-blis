##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved. ##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_l1d.h
        bli_l1d_check.c
        bli_l1d_check.h
        bli_l1d_fpa.c
        bli_l1d_fpa.h
        bli_l1d_ft.h
        bli_l1d_oapi.c
        bli_l1d_oapi.h
        bli_l1d_oapi_ba.c
        bli_l1d_oapi_ex.c
        bli_l1d_tapi.c
        bli_l1d_tapi.h
        bli_l1d_tapi_ba.c
        bli_l1d_tapi_ex.c
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
