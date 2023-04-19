##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources(${TARGET_NAME}
    PRIVATE
        bla_cabs1.c
        bla_gbmv.c
        bla_hbmv.c
        bla_hpmv.c
        bla_hpr.c
        bla_hpr2.c
        bla_lsame.c
        bla_rot.c
        bla_rotg.c
        bla_rotm.c
        bla_rotmg.c
        bla_sbmv.c
        bla_spmv.c
        bla_spr.c
        bla_spr2.c
        bla_tbmv.c
        bla_tbsv.c
        bla_tpmv.c
        bla_tpsv.c
        bla_xerbla.c
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

#Add all subdirectories
add_subdirectory(util)



