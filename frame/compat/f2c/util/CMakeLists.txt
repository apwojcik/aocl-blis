##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bla_c_abs.c
        bla_c_div.c
        bla_d_abs.c
        bla_d_cnjg.c
        bla_d_imag.c
        bla_d_sign.c
        bla_f__cabs.c
        bla_r_abs.c
        bla_r_cnjg.c
        bla_r_imag.c
        bla_r_sign.c
        bla_z_abs.c
        bla_z_div.c
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
