##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved. ##

target_sources(${TARGET_NAME}
    PRIVATE
        bli_l0.h
        bli_l0_check.c
        bli_l0_check.h
        bli_l0_fpa.c
        bli_l0_fpa.h
        bli_l0_ft.h
        bli_l0_oapi.c
        bli_l0_oapi.h
        bli_l0_tapi.c
        bli_l0_tapi.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_subdirectory(copysc)
